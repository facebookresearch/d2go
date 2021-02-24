#!/usr/bin/env python3

import d2go.modeling.misc as d2go_misc
import mobile_cv.arch.utils.fuse_utils as fuse_utils
import torch
from d2go.config import CfgNode as CN
from d2go.export.api import PredictorExportConfig
from d2go.utils.helper import alias
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.utils.registry import Registry
from mobile_cv.arch.utils.quantize_utils import wrap_quant_subclass
from mobile_cv.predictor.api import FuncInfo
from torch import nn
from torch.quantization import DeQuantStub
from torch.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx


### ================================================================================
### CONFIG
### ================================================================================
def add_regressor_default_configs(_C):
    _C.MODEL.REGRESSOR = CN()

    # head configs
    _C.MODEL.REGRESSOR.TRAIN_HEAD_NAME = ""
    _C.MODEL.REGRESSOR.HEAD_NAMES = []

    # DEPRECATED, kept for backwards compat
    _C.MODEL.REGRESSOR.HEADS = CN(new_allowed=True)  # deprecated
    _C.MODEL.REGRESSOR.HEADS.GazeHead = CN()  # deprecated
    _C.MODEL.REGRESSOR.HEADS.GazeHead.OUTPUT_CHANNELS = 4  # deprecated

    # loss configs
    _C.MODEL.REGRESSOR.LOSSES = CN()
    _C.MODEL.REGRESSOR.LOSSES.NAME = ""

    # DEPRECATED, kept for backwards compat
    _C.MODEL.REGRESSOR.LOSSES.BETA = 0.1  # deprecated
    _C.MODEL.REGRESSOR.LOSSES.THETA = 0.1  # deprecated
    _C.MODEL.REGRESSOR.LOSSES.K = 0.1  # deprecated
    _C.MODEL.REGRESSOR.LOSSES.REDUCTION = "mean"  # deprecated

    # preprocessor
    _C.MODEL.REGRESSOR.PREPROCESSORS = CN()
    _C.MODEL.REGRESSOR.PREPROCESSORS.NAME = ""
    _C.MODEL.REGRESSOR.PREPROCESSORS.SPLIT_DIM = 1
    _C.MODEL.REGRESSOR.PREPROCESSORS.CONCAT_DIM = 0
    _C.MODEL.REGRESSOR.PREPROCESSORS.CHUNK = 2


### ================================================================================
### REGRESSOR HEADS
### ================================================================================
REGRESSOR_HEADS_REGISTRY = Registry("REGRESSOR_HEADS")


def build_regressor_heads(cfg, head_names, input_channels):
    """
    Build regressor heads defined by `cfg.MODEL.REGRESSOR.HEAD_NAMES`.
    """
    heads = torch.nn.ModuleDict()
    for head_name in head_names:
        heads[head_name] = REGRESSOR_HEADS_REGISTRY.get(head_name)(cfg, input_channels)
    return heads


### ================================================================================
### REGRESSOR LOSSES
### ================================================================================
REGRESSOR_LOSS_REGISTRY = Registry("REGRESSOR_LOSSES")


def build_regressor_loss_func(cfg):
    name = cfg.MODEL.REGRESSOR.LOSSES.NAME
    return REGRESSOR_LOSS_REGISTRY.get(name)(cfg)


### ================================================================================
### PREPROCESSING
### ================================================================================
PREPROCESSORS_REGISTRY = Registry("PREPROCESSORS")


def build_preprocessor(cfg):
    """
    Build regressor heads defined by `cfg.MODEL.PREPROCESSORS.NAME`.
    """
    if not cfg.MODEL.REGRESSOR.PREPROCESSORS:
        return None

    name = cfg.MODEL.REGRESSOR.PREPROCESSORS.NAME
    return PREPROCESSORS_REGISTRY.get(name)(cfg)


@PREPROCESSORS_REGISTRY.register()
class SplitAndConcat(d2go_misc.SplitAndConcat):
    def __init__(self, cfg):
        split_dim = cfg.MODEL.REGRESSOR.PREPROCESSORS.SPLIT_DIM
        concat_dim = cfg.MODEL.REGRESSOR.PREPROCESSORS.CONCAT_DIM
        chunk = cfg.MODEL.REGRESSOR.PREPROCESSORS.CHUNK
        super(SplitAndConcat, self).__init__(split_dim, concat_dim, chunk)


### ================================================================================
### META_ARCH
### ================================================================================
@META_ARCH_REGISTRY.register()
class Regressor(nn.Module):
    """
    Regressor Arch:
        Takes input as image and returns vector of values
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.preprocessor = build_preprocessor(cfg)
        self.backbone = build_backbone(cfg)
        self.head_names = cfg.MODEL.REGRESSOR.HEAD_NAMES
        self.heads = build_regressor_heads(
            cfg, self.head_names, self.backbone.output_shape()["trunk3"].channels
        )
        self.train_head_name = cfg.MODEL.REGRESSOR.TRAIN_HEAD_NAME
        # TODO: determine whether we should be calling loss_func or combine with head
        self.loss_func = build_regressor_loss_func(cfg)
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            TODO: should we handle inputs the same way as detection where it is a
            list of items or just pass in tensors that are prebatched
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * target (optional): ground truth

                Other information that's included in the original dicts, such as:

                * index, subject_id, filename, sequence

        Returns:
            dict:
                Each key contains the loss output for one input image.
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        # TODO: Change the other GazeHead to pass in batched_inputs always
        outputs = self.heads[self.train_head_name](features["trunk3"], batched_inputs)
        # check that the target is in the input
        assert "target" in batched_inputs[0]
        targets = self.preprocess_target(batched_inputs)
        losses = self.loss_func(outputs, targets)
        return losses

    def inference(self, batched_inputs):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`

        Returns:
            list(dict)
                Each item correspond to a different image and the result
        """
        assert not self.training

        batch_size = len(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        results = {}
        for head in self.heads.values():
            head_result = head(features["trunk3"])
            if isinstance(head_result, dict):
                # Flatten all head outputs into a single dict
                for name in head.output_names:
                    results[name] = head_result[name]
            else:
                # support heads without a dict output
                results[head.output_names[0]] = head_result

        # convert results to list of dict for each head
        # assume there is a result for every head for every input
        return [
            {key: value[index] for key, value in results.items()}
            for index in range(batch_size)
        ]

    def preprocess_image(self, batched_inputs):
        """
        Batch the input images based
        """
        images = torch.FloatTensor(
            [x["image"].cpu().numpy() for x in batched_inputs]
        ).to(self.device)
        if self.preprocessor:
            images = self.preprocessor(images)
        return images

    def preprocess_target(self, batched_inputs):
        """
        Batch the targets
        """
        targets = torch.FloatTensor(
            [x["target"].cpu().numpy() for x in batched_inputs]
        ).to(self.device)
        return targets

    def prepare_for_export(self, cfg, inputs, export_scheme):
        """Create the model that will be run in predictor

        Preprocessing is called on the data before running the model.
            inputs (list of dicts) -> preprocessing (tensor) -> model

        The deployable model should be traceable. Post processing reshapes
        the output of the deployable model to whatever is used by d2go when
        running eval (list of dicts)

        NOTE: caffe2 is dealt with differently (e.g., large batch sizes are converted to
        batch size 1 and fed into the model) and kept here as legacy.
        """

        preprocess_info = FuncInfo.gen_func_info(
            PreprocessFunc, params={"device": str(self.device)}
        )
        preprocess_func = preprocess_info.instantiate()

        def data_gen(x):
            images = preprocess_func(x)  # [N, 2, H, W]
            if export_scheme == "caffe2":
                # NOTE: some operator "remembers" the batch size during tracing thus the
                # traced model only works with the same batch size during tracing. Therefore
                # only take the first batch here.
                images = images[:1]
                # TODO: this should be (images, ), needs to clean up caffe2 export API
                return ([images],)
            else:
                return (images,)  # [N, 2, H, W]

        if export_scheme == "caffe2":
            postprocess_func = PostprocessFunc
            run_func = RunFunc
            create_deployable_model = DeployableModel
        else:
            postprocess_func = TorchScriptPostprocessFunc
            run_func = TorchScriptRunFunc
            create_deployable_model = TorchScriptDeployableModel

        postprocess_info = FuncInfo.gen_func_info(
            postprocess_func,
            params={"ordered_head_output_names": self.get_ordered_head_output_names()},
        )

        return PredictorExportConfig(
            model=create_deployable_model(self),
            data_generator=data_gen,
            preprocess_info=preprocess_info,
            postprocess_info=postprocess_info,
            run_func_info=FuncInfo.gen_func_info(run_func, params={}),
        )

    def get_ordered_head_output_names(self):
        """Return flattend output names in the defined order by head
        The heads are defined in an array
        The outputs for each head are also defined in an array

        Thus, the flattened order is consistent
        New heads can be added to the end
        NOTE: a new output for a head will push down the order of
        all heads after that
        """
        flat = []
        for head_name in self.head_names:
            flat = flat + self.heads[head_name].output_names
        return flat

    def prepare_for_quant(self, cfg):
        """Creates model that has quant dequant ops

        Removes the dequant stubs that are by default added in FBNetV2Backbone
        and replaces this with just dequant after the head
        """
        assert (
            cfg.MODEL.BACKBONE.NAME == "FBNetV2C4Backbone"
        ), "Only support FBNet PyTorch quantization"

        model = self
        # set backend
        torch.backends.quantized.engine = cfg.QUANTIZATION.BACKEND
        qconfig = (
            torch.quantization.get_default_qat_qconfig(cfg.QUANTIZATION.BACKEND)
            if model.training
            else torch.quantization.get_default_qconfig(cfg.QUANTIZATION.BACKEND)
        )

        if cfg.QUANTIZATION.EAGER_MODE:
            # fuse
            # TODO: implment fuse_model with the new quantization api for extending fusion
            # TODO: we moved this here because graph mode quant does not work with eager
            # mode fusion (since the fused model has nn.Identity() after conv)
            # we may add the support in the future if this is needed
            model = fuse_utils.fuse_model(model, inplace=False)

            # Eager Mode Quantization
            # The goal is quantizing the input of backbone and dequantizing the output of
            # head, while keep things quantized in between. To achieve this, we use standard
            # way to insert quant/dequant to the backbone and replace the added dequant with
            # non-op, then only add dequant the head.
            model.backbone = wrap_quant_subclass(
                model.backbone, n_inputs=1, n_outputs=len(model.backbone._out_features)
            )
            for i in range(len(model.backbone.dequant_stubs.stubs)):
                model.backbone.dequant_stubs.stubs[i] = nn.Sequential()
            for k in model.heads.keys():
                model.heads[k] = DeQuantHead(model.heads[k])
            model.qconfig = qconfig
            # TODO(future diff): move the torch.quantization.prepare(...) call
            # here, to be consistent with the FX branch
        else:
            qconfig_dict = {"": qconfig}
            prepare_custom_config_dict = {
                "preserved_attributes": ["output_names"]
            }
            if model.training:
                model.backbone = prepare_qat_fx(model.backbone, qconfig_dict)
                for k in model.heads.keys():
                    model.heads[k] = prepare_qat_fx(model.heads[k], qconfig_dict, prepare_custom_config_dict=prepare_custom_config_dict)
            else:
                model.backbone = prepare_fx(model.backbone, qconfig_dict)
                for k in model.heads.keys():
                    model.heads[k] = prepare_fx(model.heads[k], qconfig_dict, prepare_custom_config_dict=prepare_custom_config_dict)

        return model

    def prepare_for_quant_convert(self, cfg):
        model = self
        if cfg.QUANTIZATION.EAGER_MODE:
            torch.quantization.convert(self, inplace=True)
        else:
            convert_custom_config_dict = {
                "preserved_attributes": ["output_names"]
            }
            model.backbone = convert_fx(model.backbone)
            for k in self.heads.keys():
                model.heads[k] = convert_fx(model.heads[k], convert_custom_config_dict=convert_custom_config_dict)
        return model

class DeQuantHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.noutputs = len(self.head.output_names)
        self.dequant_stubs = nn.ModuleList(DeQuantStub() for _ in range(self.noutputs))
        self.output_names = self.head.output_names

    def forward(self, *args):
        """Add dequant stubs to outputs"""
        outputs = self.head(*args)
        if isinstance(outputs, dict):
            assert len(outputs) == self.noutputs
            for i, (k, v) in enumerate(outputs.items()):
                outputs[k] = self.dequant_stubs[i](v)
            return outputs
        elif isinstance(outputs, torch.Tensor):
            return self.dequant_stubs[0](outputs)
        else:
            raise NotImplementedError("Only support dict or tensor output")


class DeployableModel(nn.Module):
    def __init__(self, pytorch_model):
        super().__init__()
        self.model = pytorch_model
        self.eval()

    def forward(self, tensor_inputs):
        images = tensor_inputs[0]

        if torch.onnx.is_in_onnx_export():
            images = alias(images, "data")

        # images.shape: [N, 2, H, W], where N is 1
        images = self.model.preprocessor(images)
        features = self.model.backbone(images)
        outputs = []
        for name in self.model.head_names:
            head_outputs = self.model.heads[name](features["trunk3"])
            if isinstance(head_outputs, dict):
                for output_name in self.model.heads[name].output_names:
                    outputs.append(head_outputs[output_name])
            else:
                outputs.append(head_outputs)

        if torch.onnx.is_in_onnx_export():
            names = self.model.get_ordered_head_output_names()
            assert len(outputs) == len(names)
            for i in range(len(outputs)):
                outputs[i] = alias(outputs[i], names[i])

        return outputs

    # legacy API, will be removed
    def encode_additional_info(self, predict_net, init_net):
        pass


class PreprocessFunc(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, batched_inputs):
        images = torch.FloatTensor(
            [x["image"].cpu().numpy() for x in batched_inputs]
        ).to(self.device)
        return images


class PostprocessFunc(object):
    def __init__(self, ordered_head_output_names):
        """Create format that is used by the evaluator"""
        # hard coded head as this is hardcoded in the evaluator
        self.ordered_head_output_names = ordered_head_output_names

    def __call__(self, _1, _2, outputs):
        """Wrap the output in what the evaluator uses"""
        # unpack the list and return dict of outputs
        #   [
        #     {"output_head0": output_head0, "output_head1": output_head1, ...},
        #     {"output_head0": output_head0, "output_head1": output_head1, ...},
        #     ...
        #   ]
        #
        # torchscript returns torch.Size([4]) but caffe2 returns torch.Size([1, 4])
        # so we squeeze the first dimension of the output
        batched_outputs = []
        for output in outputs:
            batched_outputs.append(
                {
                    self.ordered_head_output_names[i]: head_output.squeeze(0)
                    for i, head_output in enumerate(output)
                }
            )
        return batched_outputs


class RunFunc(object):
    def __call__(self, model, images):
        # the model only works with single batch, run each batch individually
        # returns list:
        #  [
        #    [output_head0, output_head1, ...], batch0
        #    [output_head0, output_head1, ...], batch1
        #    ...
        #  ]
        outputs = []
        for i in range(len(images)):
            x = images[i : i + 1]
            input_tensors = [x]
            output_tensors = model(input_tensors)
            outputs.append(output_tensors)
        return outputs


class TorchScriptDeployableModel(nn.Module):
    def __init__(self, pytorch_model):
        super().__init__()
        self.model = pytorch_model
        self.eval()

    def forward(self, images):
        # images.shape: [N, 2, H, W], where N is 1
        images = self.model.preprocessor(images)
        features = self.model.backbone(images)
        outputs = []
        for name in self.model.head_names:
            head_outputs = self.model.heads[name](features["trunk3"])
            if isinstance(head_outputs, dict):
                for output_name in self.model.heads[name].output_names:
                    outputs.append(head_outputs[output_name])
            else:
                outputs.append(head_outputs)
        return outputs


class TorchScriptRunFunc:
    def __call__(self, model, images):
        """Torchscript model can run batch as opposed to Caffe2 model

        Feed images directly into the model and returns list of lists
          [
            [head0_output0, head0_output1],
            [head1_output0, head1_output1],
              ...
          ]
        """
        return model(images)


class TorchScriptPostprocessFunc:
    def __init__(self, ordered_head_output_names):
        """Create format that is used by the evaluator"""
        # hard coded head as this is hardcoded in the evaluator
        self.ordered_head_output_names = ordered_head_output_names

    def __call__(self, _1, _2, outputs):
        """Wrap the output in what the evaluator uses

        outputs = [
            [head0_output0, head0_output1],
            [head1_output0, head1_output1],
                ...
        ]

        returns
        [
            {"output_head0": head0_output0, "output_head1": head1_output1, ...},
            {"output_head0": head0_output1, "output_head1": head1_output1, ...},
            ...
        ]
        """
        batch_size = len(outputs[0])
        return [
            {
                head_output_name: outputs[head_idx][batch_idx]
                for head_idx, head_output_name in enumerate(
                    self.ordered_head_output_names
                )
            }
            for batch_idx in range(batch_size)
        ]
