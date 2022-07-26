import logging

import torch.nn as nn
from d2go.config import CfgNode as CN
from d2go.export.api import PredictorExportConfig
from d2go.quantization.modeling import set_backend_and_create_qconfig
from d2go.registry.builtin import META_ARCH_REGISTRY
from detectron2.config import configurable
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.fcos import FCOS as d2_FCOS, FCOSHead
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import ImageList
from mobile_cv.arch.utils import fuse_utils
from mobile_cv.arch.utils.quantize_utils import wrap_quant_subclass
from mobile_cv.predictor.api import FuncInfo

logger = logging.getLogger(__name__)


class FCOSInferenceWrapper(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model

    def forward(self, image):
        """
        This function describes what happends during the tracing. Note that the output
        contains non-tensor, therefore the TracingAdaptedTorchscriptExport must be used in
        order to convert the output back from flattened tensors.
        """
        inputs = [{"image": image}]
        result = self.model.forward(inputs)[0]["instances"]
        return result

    @staticmethod
    class Preprocess(object):
        """
        This function describes how to covert orginal input (from the data loader)
        to the inputs used during the tracing (i.e. the inputs of forward function).
        """

        def __call__(self, batch, size_divisibility=32):
            assert len(batch) == 1, "only support single batch"

            image = batch[0]["image"]
            image_divisibility = ImageList.from_tensors([image], size_divisibility)
            return image_divisibility.tensor[0].to(image.device)

    @staticmethod
    class Postprocess(object):
        def __init__(self, detector_postprocess_done_in_model=False):
            """
            Args:
                detector_postprocess_done_in_model (bool): whether `detector_postprocess`
                has already applied in the D2RCNNInferenceWrapper
            """
            self.detector_postprocess_done_in_model = detector_postprocess_done_in_model

        def __call__(self, batch, inputs, outputs):
            """
            This function describes how to run the predictor using exported model. Note
            that `tracing_adapter_wrapper` runs the traced model under the hood and
            behaves exactly the same as the forward function.
            """
            assert len(batch) == 1, "only support single batch"
            width, height = batch[0]["width"], batch[0]["height"]
            if self.detector_postprocess_done_in_model:
                image_shape = batch[0]["image"].shape  # chw
                if image_shape[1] != height or image_shape[2] != width:
                    raise NotImplementedError(
                        f"Image tensor (shape: {image_shape}) doesn't match the"
                        f" input width ({width}) height ({height}). Since post-process"
                        f" has been done inside the torchscript without width/height"
                        f" information, can't recover the post-processed output to "
                        f"orignail resolution."
                    )
                return [{"instances": outputs}]
            else:
                r = detector_postprocess(outputs, height, width)
                return [{"instances": r}]


def add_fcos_configs(cfg):
    cfg.MODEL.FCOS = CN()
    # the number of foreground classes.
    cfg.MODEL.FCOS.NUM_CLASSES = 80
    cfg.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.FCOS.NUM_CONVS = 4
    cfg.MODEL.FCOS.HEAD_NORM = "BN"  # use BN if need quantization

    # inference parameters
    cfg.MODEL.FCOS.SCORE_THRESH_TEST = 0.04
    cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST = 1000
    cfg.MODEL.FCOS.NMS_THRESH_TEST = 0.6

    # Focal loss parameters
    cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA = 2.0


# Re-register D2's meta-arch in D2Go with updated APIs
@META_ARCH_REGISTRY.register()
class FCOS(d2_FCOS):
    """
    Implement config->argument translation for FCOS model.
    """

    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_for_export(self, cfg, inputs, predictor_type):

        preprocess_info = FuncInfo.gen_func_info(
            FCOSInferenceWrapper.Preprocess, params={}
        )
        preprocess_func = preprocess_info.instantiate()

        postprocess_info = FuncInfo.gen_func_info(
            FCOSInferenceWrapper.Postprocess,
            params={},
        )

        return PredictorExportConfig(
            model=FCOSInferenceWrapper(self),
            data_generator=lambda x: (preprocess_func(x),),
            model_export_method=predictor_type,  # check this
            preprocess_info=preprocess_info,
            postprocess_info=postprocess_info,
        )

    def prepare_for_quant(self, cfg, example_input=None):

        model = self
        qconfig = set_backend_and_create_qconfig(cfg, is_train=model.training)

        if cfg.QUANTIZATION.EAGER_MODE:

            model.backbone.qconfig = qconfig
            model.head.qconfig = qconfig
            model.backbone = wrap_quant_subclass(
                model.backbone,
                n_inputs=1,
                n_outputs=len(model.backbone._out_features),
            )

            model.head = wrap_quant_subclass(
                model.head,
                n_inputs=len(cfg.MODEL.FCOS.IN_FEATURES),
                n_outputs=len(cfg.MODEL.FCOS.IN_FEATURES)
                * 3,  # predictions: conf, box, clf
            )

            model = fuse_utils.fuse_model(
                model,
                is_qat=cfg.QUANTIZATION.QAT.ENABLED,
                inplace=True,
            )

        else:  # FX graph mode quantization
            raise NotImplementedError("FX mode not implemented for Yolox")

        return model

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        try:
            feature_shapes = [backbone_shape[f] for f in cfg.MODEL.FCOS.IN_FEATURES]
        except KeyError:
            raise KeyError(
                f"Available keys: {backbone_shape.keys()}.  Requested keys: {cfg.MODEL.FCOS.IN_FEATURES}"
            )
        head = FCOSHead(
            input_shape=feature_shapes,
            num_classes=cfg.MODEL.FCOS.NUM_CLASSES,
            conv_dims=[feature_shapes[0].channels] * cfg.MODEL.FCOS.NUM_CONVS,
            norm=cfg.MODEL.FCOS.HEAD_NORM,
        )
        return {
            "backbone": backbone,
            "head": head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.FCOS.NUM_CLASSES,
            "head_in_features": cfg.MODEL.FCOS.IN_FEATURES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.FCOS.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.FCOS.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }
