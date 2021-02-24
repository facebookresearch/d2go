#!/usr/bin/env python3

import logging
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.export.caffe2_modeling import (
    Caffe2MetaArch,
    convert_batched_inputs_to_c2_format,
)
from d2go.export.api import PredictorExportConfig
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.roi_heads.keypoint_head import BaseKeypointRCNNHead
from detectron2.structures import ImageList, Instances, Boxes
from d2go.config import CfgNode as CN
from mobile_cv.predictor.api import FuncInfo

logger = logging.getLogger(__name__)


class IdentityKeypointRCNNHead(BaseKeypointRCNNHead):
    """
    A dummy KeypointRCNNHead, used for computing loss and doing inference directly
    using computed heatmap.
    """
    def layers(self, x):
        return x


@META_ARCH_REGISTRY.register()
class NimbleJLP(nn.Module):
    """
    This meta arch uses high level API to create the handtracking jlp model, can
    be used to train the JLP model in D2Go.
    """

    @classmethod
    def add_default_configs(cls, _C):
        _C.MODEL.NIMBLE_JLP = CN()
        _C.MODEL.NIMBLE_JLP.INPUT_SIZE = ()  # h, w
        _C.MODEL.NIMBLE_JLP.HEATMAP_SIZE = ()  # h, w
        _C.MODEL.NIMBLE_JLP.TARGET_CHANNELS = ["heatmap", "heatmap_distance"]
        _C.MODEL.NIMBLE_JLP.HEATMAP_MODULE = ""

        # weather to use the standard loss of RoIKeypointHead
        _C.MODEL.NIMBLE_JLP.USE_KRCNN_LOSS = True
        # weather to use the loss from build_model_criterion
        _C.MODEL.NIMBLE_JLP.USE_CRITERION = False

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        assert len(cfg.MODEL.NIMBLE_JLP.INPUT_SIZE) == 2
        assert len(cfg.MODEL.NIMBLE_JLP.HEATMAP_SIZE) == 2
        self.input_size = cfg.MODEL.NIMBLE_JLP.INPUT_SIZE
        self.heatmap_size = cfg.MODEL.NIMBLE_JLP.HEATMAP_SIZE
        self.target_channels = cfg.MODEL.NIMBLE_JLP.TARGET_CHANNELS

        # NOTE: delayed import
        from handtracking.common import model_cfg
        from handtracking.jlp import build_model_criterion as bmc

        bmc.constants.num_features = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS

        # TODO: switch to FBNetV2
        fbnet_config = model_cfg.FBNetConfig(
            BN_TYPE=cfg.MODEL.FBNET.BN_TYPE,
            SCALE_FACTOR=cfg.MODEL.FBNET.SCALE_FACTOR,
            ARCH=cfg.MODEL.FBNET.ARCH,
            DW_CONV_SKIP_BNRELU=True,
        )
        arch_cfgs = model_cfg.ModelConfig(
            FBNET=fbnet_config,
            MODEL_HEAD=model_cfg.ModelHeadConfig(
                HEATMAP_SIZE=self.heatmap_size,
                HEATMAP=cfg.MODEL.NIMBLE_JLP.HEATMAP_MODULE,
            ),
        )
        model, criterion = bmc.build_model_criterion(
            "fbnet",  # TODO: make this configurable if supporing resnet
            self.target_channels,
            i_planes=3,  # NOTE: switch back to 1? needs to customize dataset mapper
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
            merge_depth=2,
            arch_cfgs=arch_cfgs,
            multi_frame=False,
        )
        self.backbone = model
        # TODO: set size_divisibility correctly
        self.backbone.size_divisibility = 0
        self.criterion = criterion if cfg.MODEL.NIMBLE_JLP.USE_CRITERION else None

        self.use_krcnn_loss = cfg.MODEL.NIMBLE_JLP.USE_KRCNN_LOSS
        self.identity_krcnn_head = IdentityKeypointRCNNHead(cfg, input_shape=None)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))  # noqa
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))  # noqa

        self.to(self.device)

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        # since we rename "self.model" to "self.backbone", rename the state_dict
        old_prefix = "model."
        new_prefix = "backbone."
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(old_prefix):
                new_k = "{}{}".format(new_prefix, k[len(old_prefix):])
                logger.warning("Rename {} to {}".format(k, new_k))
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v

        state_dict.clear()
        state_dict.update(new_state_dict)
        return super()._load_from_state_dict(state_dict, *args, **kwargs)

    def forward(self, batched_inputs):

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, size_divisibility=0)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = [None] * len(batched_inputs)

        # NOTE: add note here
        if images.tensor.shape[-2:] != torch.Size(self.input_size):
            images.tensor = F.interpolate(images.tensor, size=self.input_size)
        assert images.tensor.shape[-2:] == torch.Size(self.input_size)

        output = self.backbone([images.tensor])
        output_channels = dict(zip(self.target_channels, output))

        instances = []
        for instances_per_image, image_size in zip(gt_instances, images.image_sizes):
            height, width = image_size
            full_image_boxes = Boxes([[0, 0, width, height]]).to(self.device)

            if self.training:
                instances_per_image.proposal_boxes = full_image_boxes
                instances.append(instances_per_image)
            else:
                instances_per_image = Instances(image_size)
                # NOTE: use hardcoded scores and classes
                scores = torch.Tensor([1.0]).to(self.device)
                classes = torch.Tensor([0]).to(torch.int64).to(self.device)
                instances_per_image.pred_boxes = full_image_boxes
                instances_per_image.scores = scores
                instances_per_image.pred_classes = classes
                instances.append(instances_per_image)

        if self.training:
            loss = {}
            if self.use_krcnn_loss:
                loss.update(
                    self.identity_krcnn_head(output_channels["heatmap"], instances)
                )
            if self.criterion:
                # TODO: genearte the target and then call:
                # total_loss, sub_loss = self.criterion(output, target)
                raise NotImplementedError()
            else:
                assert self.use_krcnn_loss and self.target_channels == ["heatmap"]
            return loss
        else:
            self.identity_krcnn_head(output_channels["heatmap"], instances)
            return GeneralizedRCNN._postprocess(
                instances, batched_inputs, images.image_sizes
            )

    def prepare_for_export(self, cfg, inputs, export_scheme):
        preprocess_func_info = FuncInfo.gen_func_info(PreprocessFunc, params={
            "size_divisibility": self.backbone.size_divisibility,
            "device": str(self.device),
        })
        postprocess_func_info = FuncInfo.gen_func_info(PostprocessFunc, params={})
        preprocess_func = preprocess_func_info.instantiate()

        return PredictorExportConfig(
            model=JLPCaffe2MetaArch(cfg, self),
            data_generator=lambda x: (preprocess_func(x), ),
            preprocess_info=preprocess_func_info,
            postprocess_info=postprocess_func_info,
        )


class JLPCaffe2MetaArch(Caffe2MetaArch):
    def encode_additional_info(self, predict_net, init_net):
        pass

    # @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        data = inputs[0]
        unused_im_info = torch.Tensor()
        images = self._caffe2_preprocess_image((data, unused_im_info))
        # NOTE: add note here
        if images.tensor.shape[-2:] != torch.Size(self._wrapped_model.input_size):
            images.tensor = F.interpolate(images.tensor, size=self._wrapped_model.input_size)
        assert images.tensor.shape[-2:] == torch.Size(self._wrapped_model.input_size)

        outputs = self._wrapped_model.backbone([images.tensor])
        return tuple(outputs)

    @staticmethod
    def get_outputs_converter(predict_net, init_net):
        def f(batched_inputs, c2_inputs, c2_results):
            raise NotImplementedError()

        return f


class PreprocessFunc(object):
    def __init__(self, size_divisibility, device):
        self.size_divisibility = size_divisibility
        self.device = device

    def __call__(self, inputs):
        data, im_info = convert_batched_inputs_to_c2_format(
            inputs, self.size_divisibility, self.device
        )
        # NOTE: the model only takes a single image tensor as input
        return (data, )


class PostprocessFunc(object):
    # NOTE: no parameter for post-process, thus no __init__

    def __call__(self, inputs, tensor_inputs, tensor_outputs):
        # construnct "fake" instances
        instances = []
        image_sizes = [(x["height"], x["width"]) for x in inputs]
        for image_size in image_sizes:
            device = tensor_outputs[0].device
            height, width = image_size
            full_image_boxes = Boxes([[0, 0, width, height]]).to(device)

            instances_per_image = Instances(image_size)
            # NOTE: use hardcoded scores and classes
            scores = torch.Tensor([1.0]).to(device)
            classes = torch.Tensor([0]).to(torch.int64).to(device)
            instances_per_image.pred_boxes = full_image_boxes
            instances_per_image.scores = scores
            instances_per_image.pred_classes = classes
            instances.append(instances_per_image)

        from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_inference
        x = tensor_outputs[0]
        keypoint_rcnn_inference(x, instances)

        return GeneralizedRCNN._postprocess(instances, inputs, image_sizes)
