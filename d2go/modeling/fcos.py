from d2go.config import CfgNode as CN
from detectron2.config import configurable
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.fcos import FCOS as d2_FCOS, FCOSHead


def add_fcos_configs(cfg):
    cfg.MODEL.FCOS = CN()
    # the number of foreground classes.
    cfg.MODEL.FCOS.NUM_CLASSES = 80
    cfg.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.FCOS.NUM_CONVS = 4
    cfg.MODEL.FCOS.HEAD_NORM = "GN"

    # inference parameters
    cfg.MODEL.FCOS.SCORE_THRESH_TEST = 0.04
    cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST = 1000
    cfg.MODEL.FCOS.NMS_THRESH_TEST = 0.6

    # Focal loss parameters
    cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA = 2.0


@META_ARCH_REGISTRY.register()
class FCOS(d2_FCOS):
    """
    Implement config->argument translation for FCOS model.
    """

    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
