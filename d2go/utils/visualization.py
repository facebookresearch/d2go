#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Optional, Type

from d2go.config import CfgNode as CN
from d2go.registry.builtin import META_ARCH_REGISTRY
from detectron2.data import DatasetCatalog, detection_utils as utils, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer


def add_tensorboard_default_configs(_C):
    _C.TENSORBOARD = CN()
    # Output from dataloader will be written to tensorboard at this frequency
    _C.TENSORBOARD.TRAIN_LOADER_VIS_WRITE_PERIOD = 20
    # This controls max number of images over all batches, be considerate when
    # increasing this number because it takes disk space and slows down the training
    _C.TENSORBOARD.TRAIN_LOADER_VIS_MAX_IMAGES = 16
    # This controls the max number of images to visualize each write period
    _C.TENSORBOARD.TRAIN_LOADER_VIS_MAX_BATCH_IMAGES = 16
    # Max number of images per dataset to visualize in tensorboard during evaluation
    _C.TENSORBOARD.TEST_VIS_MAX_IMAGES = 16
    # Frequency of sending data to tensorboard during evaluation
    _C.TENSORBOARD.TEST_VIS_WRITE_PERIOD = 1

    # TENSORBOARD.LOG_DIR will be determined solely by OUTPUT_DIR
    _C.register_deprecated_key("TENSORBOARD.LOG_DIR")

    return _C


class VisualizerWrapper(object):
    """
    D2's Visualizer provides low-level APIs to draw common structures, such as
    draw_instance_predictions/draw_sem_seg/overlay_instances. This class provides
    the high-level interface for visualizing.
    """

    def __init__(self, cfg, custom_visualizer: Optional[Type[Visualizer]] = None):
        self.cfg = cfg
        self.visualizer = custom_visualizer or Visualizer

    def _get_meta_arch_class(self):
        return META_ARCH_REGISTRY.get(self.cfg.MODEL.META_ARCHITECTURE)

    def visualize_train_input(self, input_dict):
        """
        Visulize a single input image of model (also the output from train loader)
        used for training, this contains the data augmentation.
        """
        per_image = input_dict
        cfg = self.cfg

        # customization
        if hasattr(self._get_meta_arch_class(), "visualize_train_input"):
            return self._get_meta_arch_class().visualize_train_input(self, input_dict)

        img = per_image["image"].permute(1, 2, 0).detach().cpu().numpy()
        img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

        if "dataset_name" in input_dict:
            metadata = MetadataCatalog.get(input_dict["dataset_name"])
        else:
            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        scale = 2.0
        visualizer = self.visualizer(img, metadata=metadata, scale=scale)

        if "instances" in per_image:
            target_fields = per_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )

        if "sem_seg" in per_image:
            visualizer.draw_sem_seg(per_image["sem_seg"], area_threshold=0, alpha=0.5)

        return visualizer.get_output().get_image()

    def visualize_test_output(
        self, dataset_name, dataset_mapper, input_dict, output_dict
    ):
        """
        Visualize the output of model
        """

        # customization
        if hasattr(self._get_meta_arch_class(), "visualize_test_output"):
            return self._get_meta_arch_class().visualize_test_output(
                self, dataset_name, dataset_mapper, input_dict, output_dict
            )

        image = dataset_mapper._read_image(input_dict, "RGB")
        visualizer = self.visualizer(image, metadata=MetadataCatalog.get(dataset_name))

        if "panoptic_seg" in output_dict:
            panoptic_seg, segments_info = output_dict["panoptic_seg"]
            visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to("cpu"), segments_info
            )
        if "instances" in output_dict:
            visualizer.draw_instance_predictions(output_dict["instances"].to("cpu"))
        if "sem_seg" in output_dict:
            visualizer.draw_sem_seg(
                output_dict["sem_seg"].argmax(dim=0).to("cpu"),
                area_threshold=0,
                alpha=0.5,
            )

        return visualizer.get_output().get_image()

    def visualize_dataset_dict(self, dataset_name, dataset_mapper, dataset_dict):
        """
        Visualize the dataset_dict
        """
        image = dataset_mapper._read_image(dataset_dict, "RGB")
        visualizer = self.visualizer(image, metadata=MetadataCatalog.get(dataset_name))
        visualizer.draw_dataset_dict(dataset_dict)
        return visualizer.get_output().get_image()


class DataLoaderVisWrapper:
    """
    Wrap the data loader to visualize its output via TensorBoardX at given frequency.
    """

    def __init__(
        self,
        cfg,
        tbx_writer,
        data_loader,
        visualizer: Optional[Type[VisualizerWrapper]] = None,
    ):
        self.tbx_writer = tbx_writer
        self.data_loader = data_loader
        self._visualizer = visualizer(cfg) if visualizer else VisualizerWrapper(cfg)

        self.log_frequency = cfg.TENSORBOARD.TRAIN_LOADER_VIS_WRITE_PERIOD
        self.log_limit = cfg.TENSORBOARD.TRAIN_LOADER_VIS_MAX_IMAGES
        self.batch_log_limit = cfg.TENSORBOARD.TRAIN_LOADER_VIS_MAX_BATCH_IMAGES
        assert self.log_frequency >= 0
        assert self.log_limit >= 0
        assert self.batch_log_limit >= 0
        self._remaining = self.log_limit

    def __iter__(self):
        for data in self.data_loader:
            self._maybe_write_vis(data)
            yield data

    def _maybe_write_vis(self, data):
        try:
            storage = get_event_storage()
        except AssertionError:
            # wrapped data loader might be used outside EventStorage, don't visualize
            # anything
            return

        if (
            self.log_frequency == 0
            or not storage.iter % self.log_frequency == 0
            or self._remaining <= 0
        ):
            return

        length = min(len(data), min(self.batch_log_limit, self._remaining))
        data = data[:length]
        self._remaining -= length

        for i, per_image in enumerate(data):
            vis_image = self._visualizer.visualize_train_input(per_image)
            tag = [f"train_loader_batch_{storage.iter}"]
            if "dataset_name" in per_image:
                tag += [per_image["dataset_name"]]
            if "file_name" in per_image:
                tag += [f"img_{i}", per_image["file_name"]]

            if isinstance(vis_image, dict):
                for k in vis_image:
                    self.tbx_writer._writer.add_image(
                        tag="/".join(tag + [k]),
                        img_tensor=vis_image[k],
                        global_step=storage.iter,
                        dataformats="HWC",
                    )
            else:
                self.tbx_writer._writer.add_image(
                    tag="/".join(tag),
                    img_tensor=vis_image,
                    global_step=storage.iter,
                    dataformats="HWC",
                )


class VisualizationEvaluator(DatasetEvaluator):
    """
    Visualize GT and prediction during evaluation. It doesn't calculate any
        metrics, just uses evaluator's interface as hook.
    """

    # NOTE: the evaluator will be created for every eval (during training and
    # after training), so the images will be logged multiple times, use a global
    # counter to differentiate them in TB.
    _counter = 0

    def __init__(
        self,
        cfg,
        tbx_writer,
        dataset_mapper,
        dataset_name,
        train_iter=None,
        tag_postfix=None,
        visualizer: Optional[Type[VisualizerWrapper]] = None,
    ):
        self.tbx_writer = tbx_writer
        self.dataset_mapper = dataset_mapper
        self.dataset_name = dataset_name
        self._visualizer = visualizer(cfg) if visualizer else VisualizerWrapper(cfg)
        self.train_iter = train_iter or VisualizationEvaluator._counter
        self.tag_postfix = tag_postfix or ""

        self.log_limit = max(cfg.TENSORBOARD.TEST_VIS_MAX_IMAGES, 0)
        self.log_frequency = cfg.TENSORBOARD.TEST_VIS_WRITE_PERIOD

        self._metadata = None
        self._dataset_dict = None
        self._file_name_to_dataset_dict = None
        if self.log_limit > 0:
            self._initialize_dataset_dict(dataset_name)

        VisualizationEvaluator._counter += 1
        self.reset()

    def _initialize_dataset_dict(self, dataset_name: str) -> None:
        # Enable overriding defaults in case the dataset hasn't been registered.

        self._metadata = MetadataCatalog.get(dataset_name)
        # NOTE: Since there's no GT from test loader, we need to get GT from
        # the dataset_dict, this assumes the test data loader uses the item from
        # dataset_dict in the default way.
        self._dataset_dict = DatasetCatalog.get(dataset_name)
        self._file_name_to_dataset_dict = {
            dic["file_name"]: dic for dic in self._dataset_dict
        }

    def reset(self):
        self._iter = 0
        self._log_remaining = self.log_limit

    def process(self, inputs, outputs):
        if (
            self.log_frequency == 0
            or self._iter % self.log_frequency != 0
            or self._log_remaining <= 0
        ):
            self._iter += 1
            return

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            dataset_dict = self._file_name_to_dataset_dict[file_name]
            gt_img = self._visualizer.visualize_dataset_dict(
                self.dataset_name, self.dataset_mapper, dataset_dict
            )
            pred_img = self._visualizer.visualize_test_output(
                self.dataset_name, self.dataset_mapper, input, output
            )

            tag_base = f"{self.dataset_name}{self.tag_postfix}/eval_iter_{self._iter}/{file_name}"
            self.tbx_writer._writer.add_image(
                f"{tag_base}/GT",
                gt_img,
                self.train_iter,
                dataformats="HWC",
            )

            if not isinstance(pred_img, dict):
                pred_img = {"Pred": pred_img}

            for img_type in pred_img.keys():
                self.tbx_writer._writer.add_image(
                    f"{tag_base}/{img_type}",
                    pred_img[img_type],
                    self.train_iter,
                    dataformats="HWC",
                )

            self._log_remaining -= 1

        self._iter += 1

    def has_finished_process(self):
        return True
