#!/usr/bin/env python3

import copy
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from d2go.export.api import PredictorExportConfig
from d2go.modeling.trimap import generate_weight_mask
from detectron2.data import MetadataCatalog, detection_utils as utils
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.modeling.meta_arch.semantic_seg import (
    SEM_SEG_HEADS_REGISTRY,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.visualizer import Visualizer
from fvcore.common.file_io import PathManager
from mobile_cv.predictor.api import FuncInfo
from torch.nn import functional as F


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class TemporalSemanticSegmentor(nn.Module):
    """
    Temperal extension for SemanticSegmentor. The input of this meta arch will be
    a batch of:
    {
        "filename/width/height/...": ...,
        "frames": {
            "frame_1": {"image": chw_tensor, ...},
            "frame_2": {"image": chw_tensor, ...},
            ...
        },
    }
    And the output of this meta arch will be a batch of:
    {
        "result_1": {"sem_seg": chw_tensor},
        "result_2": {"sem_seg": chw_tensor},
        ...
    }
    Currently it requires #results match #frames.
    """

    def __init__(self, cfg):
        super().__init__()
        in_channels = len(cfg.MODEL.PIXEL_MEAN)
        self.backbone = build_backbone(cfg, input_shape=ShapeSpec(channels=in_channels))
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )  # noqa
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )  # noqa
        self.preprocess = PreprocessFunc(
            size_divisibility=self.backbone.size_divisibility,
            device=cfg.MODEL.DEVICE,
        )
        self.postprocess = PostprocessFunc(
            super_classes=self.sem_seg_head._super_classes,
            superclass_id_to_channel_ids=self.sem_seg_head._superclass_id_to_channel_ids,
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def _get_targets(self, batched_inputs):
        frame_names = _get_frame_names(batched_inputs)
        if "sem_seg" in batched_inputs[0]["frames"][frame_names[0]]:
            targets_per_frame = {}
            for k in frame_names:
                targets = [
                    x["frames"][k]["sem_seg"].to(self.device) for x in batched_inputs
                ]
                targets = ImageList.from_tensors(
                    targets,
                    self.backbone.size_divisibility,
                    self.sem_seg_head.ignore_value,
                ).tensor
                targets_per_frame[k] = targets
            return targets_per_frame
        else:
            return None

    def forward(self, batched_inputs):
        frame_names = _get_frame_names(batched_inputs)
        images = self.preprocess(batched_inputs)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        features = self.backbone(dict(zip(frame_names, images)))
        assert set(features.keys()) == set(
            frame_names
        ), "Keys of the features ({}) and frame_names ({}) doesn't match".format(
            features.keys(), frame_names
        )

        targets = self._get_targets(batched_inputs)
        results, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses

        # NOTE: For single-image based SemanticSegmentor, "results" is a single
        # 4D tensor, here although using the same head registery, "results" should
        # be a dict of 4D tensors potentially representing results for different
        # frames.
        assert len(results) == len(_get_frame_names(batched_inputs))
        processed_results = self.postprocess(batched_inputs, images, results.values())
        return processed_results

    def prepare_for_quant(self, cfg):
        if hasattr(self.backbone, "delegate_prepare_for_quant"):
            self.backbone = self.backbone.delegate_prepare_for_quant(cfg)
            # Assumes all computation is done in backbone
            return self
        else:
            raise RuntimeError(
                "It seems the backbone doesn't implement delegate_prepare_for_quant,"
                " quantization might fail."
            )

    def prepare_for_quant_convert(self, cfg):
        if hasattr(self.backbone, "delegate_prepare_for_quant_convert"):
            self.backbone = self.backbone.delegate_prepare_for_quant_convert(cfg)
            # Assumes all computation is done in backbone
            return self
        else:
            raise RuntimeError(
                "It seems the backbone doesn't implement delegate_prepare_for_quant_convert,"
                " quantization might fail."
            )

    def prepare_for_export(self, cfg, inputs, export_scheme):
        preprocess_params = {
            "size_divisibility": self.backbone.size_divisibility,
            "device": cfg.MODEL.DEVICE,
        }
        preprocess_info = FuncInfo.gen_func_info(
            PreprocessFunc, params=preprocess_params
        )
        postprocess_info = FuncInfo.gen_func_info(
            PostprocessFunc,
            params={
                "super_classes": self.sem_seg_head._super_classes,
                "superclass_id_to_channel_ids": self.sem_seg_head._superclass_id_to_channel_ids,
            },
        )

        # Since different backbone may needs to be exported in different ways, delegate
        # the prepare_for_export to backbone.
        if hasattr(self.backbone, "delegate_prepare_for_export"):
            # NOTE: For simplicity, we assume the sem_seg_head has no "real" computation
            # (eg. conv layer) during inference and delegate the export to backbone.
            # The sem_seg_head can still contain non-computing operation such as
            # re-arrange the data. Here we check the state dict to falsify sem_seg_head
            # with potential compuatation.
            assert (
                len(self.sem_seg_head.state_dict()) == 0
            ), "The sem_seg_head {} might contain computation, see comments above".format(
                self.sem_seg_head
            )

            export_config = self.backbone.delegate_prepare_for_export(
                cfg,
                inputs,
                meta_arch_pixel_mean=self.pixel_mean,
                meta_arch_pixel_std=self.pixel_std,
                meta_arch_preprocess_info=preprocess_info,
                meta_arch_postprocess_info=postprocess_info,
            )
            return export_config

        # Otherwise simply trace the entire model

        frame_names = _get_frame_names(inputs)

        class Caffe2MetaArch(nn.Module):
            def __init__(self, model):
                super().__init__()
                self._wrapped_model = model
                self.training = model.training

            def forward(self, tensors):
                tensors = [
                    (x - self._wrapped_model.pixel_mean) / self._wrapped_model.pixel_std
                    for x in tensors
                ]
                features = self._wrapped_model.backbone(
                    {k: v for k, v in zip(frame_names, tensors)}
                )
                results, _ = self._wrapped_model.sem_seg_head(features, None)
                return tuple(results.values())

            def encode_additional_info(self, predict_net, init_net):
                # NOTE: current export mechanism relies this method to encode
                # preprocessing in caffe2 model. Here we use PreprocessFunc instead
                # which can also work with torchscript.
                pass

        preprocess_func = preprocess_info.instantiate()

        return PredictorExportConfig(
            model=Caffe2MetaArch(self),
            data_generator=lambda x: (preprocess_func(x),),
            preprocess_info=preprocess_info,
            postprocess_info=postprocess_info,
        )

    def export_predictor(self, cfg, predictor_type, output_dir, data_loader):
        import os

        from d2go.export.api import default_export_predictor
        from d2go.projects.person_segmentation.publish import (
            SegmentationWrapper,
            optimize_and_save_scripted_model,
        )

        predictor_path = default_export_predictor(
            cfg, self, predictor_type, output_dir, data_loader
        )

        """ Adding customized handling for exporting lite_interpreter """

        if (
            "torchscript" in predictor_type
            and cfg.MODEL.BACKBONE.NAME
            == "FBNetV2HRDetectWithMask"  # TODO: support other backbone
        ):
            # manual parameters
            # One of ["segmentation", "hair_segmentation", "multiclass_segmentation"]
            name = ""
            # AR Delivery version (https://www.fburl.com/wiki/99ocdaoa).
            version = -1
            # One of [0, 1], where 0 means High End and 1 means Low End.
            tier = -1

            model_tag = f"{name}_v{version}_t{tier}"
            outfile = os.path.join(predictor_path, f"{model_tag}.pb")

            # derived parameters
            if "ResizeShortestEdgeSquareOp" in cfg.D2GO_DATA.AUG_OPS.TEST:
                spatial_dim = (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST)
                input_type = 1  # defined in D23725583
            elif "ResizeShortestEdgeOp" in cfg.D2GO_DATA.AUG_OPS.TEST:
                min_size = cfg.INPUT.MIN_SIZE_TEST
                max_size = cfg.INPUT.MAX_SIZE_TEST
                spatial_dim = (min_size, max_size)
                input_type = 2  # defined in D23725583
            else:
                raise NotImplementedError(
                    "Model type is not supported (unknown D2GO_DATA.AUG_OPS.TEST)."
                )

            # detmask models
            if cfg.MODEL.BACKBONE.NAME == "FBNetV2HRDetectWithMask":
                input_type += 2  # defined in D23725583
                is_track = True
            else:
                raise NotImplementedError("DetectTrack models is not supported yet.")
            is_multiclass = False
            fake_input = [
                (torch.zeros(1, 4 if is_track else 3, spatial_dim[0], spatial_dim[1]),)
            ]
            metadata = (
                int(spatial_dim[0]),
                int(spatial_dim[1]),
                str(name),
                str(version),
                int(input_type),
            )

            with PathManager.open(os.path.join(predictor_path, "model.jit"), "rb") as f:
                traced_ts_model = torch.jit.load(f)
            wrapped_model = SegmentationWrapper(
                model=traced_ts_model,
                is_multiclass=is_multiclass,
                metadata=metadata,
            )
            # Script the wrapper.
            swrap = torch.jit.script(wrapped_model)
            optimize_and_save_scripted_model(
                scripted_model=swrap,
                tag=model_tag,
                sample_input=fake_input,
                outfile=outfile,
            )

        return predictor_path

    @staticmethod
    def visualize_train_input(visualizer_wrapper, input_dict):
        per_image = input_dict
        cfg = visualizer_wrapper.cfg

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        scale = 2.0

        vis_images = []
        for frame_name, frame in per_image["frames"].items():
            img = frame["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
            if metadata.get("mcs_metadata", None) is not None:
                assert set(frame["super_classes"]) <= set(metadata.mcs_metadata.keys())
                mcs_images = []
                for class_name, sem_seg in zip(
                    frame["super_classes"], frame["sem_seg"]
                ):
                    metadata_ = copy.deepcopy(metadata)
                    metadata_.set(**metadata.mcs_metadata[class_name])
                    visualizer = Visualizer(img, metadata=metadata_, scale=scale)
                    visualizer.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
                    visualizer.draw_text(
                        "{}-{}".format(frame_name, class_name),
                        (0, 0),
                        horizontal_alignment="left",
                    )
                    mcs_images.append(visualizer.get_output().get_image())
                # concat all MCS classes vertically
                vis_images.append(np.concatenate(mcs_images, axis=0))
            else:
                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                visualizer.draw_sem_seg(frame["sem_seg"], area_threshold=0, alpha=0.5)
                visualizer.draw_text(frame_name, (0, 0), horizontal_alignment="left")
                vis_images.append(visualizer.get_output().get_image())

        # putting all images side-by-side horizontally
        vis_img = np.concatenate(vis_images, axis=1)  # hwc
        return vis_img

    @staticmethod
    def visualize_test_output(
        visualizer_wrapper, dataset_name, dataset_mapper, input_dict, output_dict
    ):

        image = dataset_mapper._read_image(input_dict, "RGB")
        vis_images = []
        for frame_name, result_per_frame in output_dict.items():
            metadata = MetadataCatalog.get(dataset_name)

            if metadata.get("mcs_metadata", None) is not None:
                mcs_images = []
                for class_name, sem_seg in result_per_frame["sem_seg"].items():
                    metadata_ = copy.deepcopy(metadata)
                    metadata_.set(**metadata.mcs_metadata[class_name])
                    visualizer = Visualizer(image, metadata=metadata_)
                    visualizer.draw_sem_seg(
                        sem_seg.argmax(dim=0).to("cpu"),
                        area_threshold=0,
                        alpha=0.5,
                    )
                    visualizer.draw_text(
                        "{}-{}".format(frame_name, class_name),
                        (0, 0),
                        horizontal_alignment="left",
                    )
                    mcs_images.append(visualizer.get_output().get_image())
                # concat all MCS classes vertically
                vis_images.append(np.concatenate(mcs_images, axis=0))
            else:
                visualizer = Visualizer(image, metadata=metadata)
                visualizer.draw_sem_seg(
                    result_per_frame["sem_seg"].argmax(dim=0).to("cpu"),
                    area_threshold=0,
                    alpha=0.5,
                )
                visualizer.draw_text(frame_name, (0, 0), horizontal_alignment="left")
                vis_images.append(visualizer.get_output().get_image())

        # putting all images side-by-side
        vis_img = np.concatenate(vis_images, axis=1)  # hwc
        return vis_img


class PreprocessFunc(object):
    def __init__(self, size_divisibility, device):
        self.size_divisibility = size_divisibility
        self.device = device

    def __call__(self, batched_inputs):
        images_per_frame = _batching_images(
            batched_inputs, self.size_divisibility, self.device
        )
        return tuple(x.tensor for x in images_per_frame.values())


class PostprocessFunc(object):
    def __init__(self, super_classes, superclass_id_to_channel_ids):
        self._super_classes = super_classes
        self._superclass_id_to_channel_ids = superclass_id_to_channel_ids

    def __call__(self, batched_inputs, input_tensors, output_tensors):
        # NOTE: get the image_sizes (without padding) for each frame
        images_per_frame = _batching_images(batched_inputs)

        processed_results = [{} for _ in batched_inputs]
        for frame_name, results_per_f in zip(images_per_frame.keys(), output_tensors):
            image_sizes = images_per_frame[frame_name].image_sizes
            for batch_i, (input_per_image, image_size) in enumerate(
                zip(batched_inputs, image_sizes)
            ):
                height = input_per_image.get("height")
                width = input_per_image.get("width")

                if len(self._superclass_id_to_channel_ids) > 1:  # muti-superclass
                    processed_results[batch_i][frame_name] = {"sem_seg": {}}
                    for superclass, ids in zip(
                        self._super_classes, self._superclass_id_to_channel_ids
                    ):
                        # results_per_f contains channels across all super classes
                        result = results_per_f[:, ids, :, :]  # nchw
                        result = _convert_binary_logits_to_two_class_logits(result)
                        r = sem_seg_postprocess(result, image_size, height, width)
                        processed_results[batch_i][frame_name]["sem_seg"][
                            superclass
                        ] = r
                else:
                    assert len(results_per_f.shape) == 4
                    result = results_per_f[batch_i]
                    result = _convert_binary_logits_to_two_class_logits(result)
                    r = sem_seg_postprocess(result, image_size, height, width)
                    processed_results[batch_i][frame_name] = {"sem_seg": r}

        return processed_results


def _get_frame_names(batched_inputs):
    frame_names = list(batched_inputs[0]["frames"].keys())
    if not all(len(x["frames"]) == len(frame_names) for x in batched_inputs):
        raise NotImplementedError(
            "Currently requires all batches having same number of frames"
        )
    return frame_names


def _batching_images(batched_inputs, size_divisibility=0, device="cpu"):
    images_per_frame = {}
    for k in _get_frame_names(batched_inputs):
        images = [x["frames"][k]["image"].to(device) for x in batched_inputs]
        images = ImageList.from_tensors(images, size_divisibility)
        images_per_frame[k] = images
    return images_per_frame


@SEM_SEG_HEADS_REGISTRY.register()
class BinarySemSegLossHead(nn.Module):
    """
    Similar to SemSegFPNHead but the input features are final bitmap (no real
    compute is done in this class). It's suitable for binary output (logits with
    single channel for foreground and supports various losses. For inference, the
    binary predictions will be converetd into multiclass predictions in order to
    fit D2's evaluation and visualization implementation.

    For binary predictions, it supports following losses:
        1. loss_sem_seg: unweighted loss.
        2. loss_sem_seg_trimap_weighted: loss with weight obtained from trimap of GT mask.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.cfg = cfg
        assert (
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 2
        ), "BinarySemSegLossHead requires 2 classes"
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        # NOTE: for binary segmentation, the ignore value (affecting the padding due to
        # different aspect ratio or size divisibility) should be 0 (background)
        self.ignore_value = 0

        self.loss_weights = {
            "loss_sem_seg": self.cfg.MODEL.SEM_SEG_HEAD.LOSS_HEAD_HYPERPARAMS.BCE_WEIGHT,
            "loss_sem_seg_trimap_weighted": self.cfg.MODEL.SEM_SEG_HEAD.LOSS_HEAD_HYPERPARAMS.BCE_TRIMAP_WEIGHT,
        }
        self.trimap_ring_weight = (
            self.cfg.MODEL.SEM_SEG_HEAD.LOSS_HEAD_HYPERPARAMS.TRIMAP_RING_WEIGHT
        )

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (predictions, {})
        """
        x = self.layers(features)
        if self.training:
            return None, self.losses(x, targets)
        else:
            # NOTE: If needed, this conversion can be skipped when exporting the model
            x = _convert_binary_logits_to_two_class_logits(x)
            return x, {}

    def layers(self, features):
        assert len(features) == 1
        x = list(features.values())[0]
        assert x.shape[1] in [1, self.num_classes], x.shape
        return x

    def losses(self, predictions, targets):
        """
        Compuate the loss.

        Args:
            predictions (Tensor): predicted mask bitmap logits in float32, it's shape
                should be (N,1,H,W) representing logits of foreground class.
            targets (Tensor): int64 tensor of shape (N,H,W) representing class id.
                For binary classfication case, the class id is 1 for foreground and
                0 for background.

        Returns:
            loss (Dict[str, Tensor]): the loss
        """
        raise NotImplementedError("BinarySemSegLossHead.losses()")


@SEM_SEG_HEADS_REGISTRY.register()
class BinarySemSegBCELossHead(BinarySemSegLossHead):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)

    def losses(self, predictions, targets):
        """
        Vanilla Binary Cross Entropy
        """
        # If the predictions are for batches of clips, then the size will be (N, F, 1, H, W),
        #   F represents the clip length. We first reshape into (N*F, 1, H, W) and then feed
        if len(predictions.shape) > 4:
            predictions = predictions.contiguous().view(
                np.prod(predictions.shape[0:-3]),
                1,
                predictions.shape[-2],
                predictions.shape[-1],
            )

        assert predictions.shape[1] == 1
        # predictions: (N, 1, H, W) -> (N, H, W)
        predictions = predictions.view(-1, predictions.shape[-2], predictions.shape[-1])
        targets = targets.to(torch.float32)
        loss = {}

        for name, weight in self.loss_weights.items():
            if not weight > 0:
                continue
            if name == "loss_sem_seg":
                loss_ = F.binary_cross_entropy_with_logits(
                    predictions, targets, reduction="mean"
                )
            if name == "loss_sem_seg_trimap_weighted":
                # NOTE: same as https://fburl.com/diffusion/2ekz1a24
                weighted_trimap = np.asarray(
                    [
                        generate_weight_mask(x, ring_weight=self.trimap_ring_weight)
                        for x in targets.cpu().numpy()
                    ]
                )
                weighted_trimap = torch.from_numpy(weighted_trimap).to(targets.device)
                loss_ = F.binary_cross_entropy_with_logits(
                    predictions, targets, reduction="mean", weight=weighted_trimap
                )
            loss[name + "_bce"] = loss_ * weight

        return loss


@SEM_SEG_HEADS_REGISTRY.register()
class BinarySemSegFocalLossHead(BinarySemSegLossHead):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self.gamma = self.cfg.MODEL.SEM_SEG_HEAD.LOSS_HEAD_HYPERPARAMS.FOCAL_LOSS_GAMMA

    def losses(self, predictions, targets):
        """
        Focal Loss (https://fburl.com/fvwirpp0)
        """

        assert predictions.shape[1] == 1
        # predictions: (N, 1, H, W) -> (N, H, W)
        predictions = predictions.view(-1, predictions.shape[-2], predictions.shape[-1])
        targets = targets.to(torch.float32)
        loss = {}

        for name, weight in self.loss_weights.items():
            if not weight > 0:
                continue
            if name == "loss_sem_seg":
                BCE_loss = F.binary_cross_entropy_with_logits(
                    predictions, targets, reduction="none"
                )
                pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
                loss_ = (1 - pt) ** self.gamma * BCE_loss
            if name == "loss_sem_seg_trimap_weighted":
                # # NOTE: same as https://fburl.com/diffusion/2ekz1a24
                weighted_trimap = np.asarray(
                    [
                        generate_weight_mask(x, ring_weight=self.trimap_ring_weight)
                        for x in targets.cpu().numpy()
                    ]
                )
                weighted_trimap = torch.from_numpy(weighted_trimap).to(targets.device)
                BCE_loss = F.binary_cross_entropy_with_logits(
                    predictions, targets, reduction="none", weight=weighted_trimap
                )
                pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
                loss_ = (1 - pt) ** self.gamma * BCE_loss
            loss[name + "_focal"] = loss_.mean() * weight

        return loss


def _convert_binary_logits_to_two_class_logits(logits):
    assert len(logits.shape) in {3, 4} # (n)chw
    if logits.shape[-3] >= 2:
        return logits
    two_class_logits = torch.cat([-logits, logits], dim=-3)  # background  # foreground
    return two_class_logits
