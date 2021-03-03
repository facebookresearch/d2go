#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from detectron2.data import DatasetCatalog, MetadataCatalog, detection_utils as utils
from detectron2.evaluation import DatasetEvaluator
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer


class VisualizerWrapper(object):
    """
    D2's Visualizer provides low-level APIs to draw common structures, such as
    draw_instance_predictions/draw_sem_seg/overlay_instances. This class provides
    the high-level interface for visualizing.
    """

    def __init__(self, cfg):
        self.cfg = cfg

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

        img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
        img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        scale = 2.0
        visualizer = Visualizer(img, metadata=metadata, scale=scale)

        if "instances" in per_image:
            target_fields = per_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )

        if "sem_seg" in per_image:
            vis = visualizer.draw_sem_seg(
                per_image["sem_seg"], area_threshold=0, alpha=0.5
            )

        return vis.get_image()

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
        visualizer = Visualizer(image, metadata=MetadataCatalog.get(dataset_name))

        if "panoptic_seg" in output_dict:
            # NOTE: refer to https://fburl.com/diffusion/evarrhbh
            raise NotImplementedError()
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
        visualizer = Visualizer(image, metadata=MetadataCatalog.get(dataset_name))
        visualizer.draw_dataset_dict(dataset_dict)
        return visualizer.get_output().get_image()


class DataLoaderVisWrapper:
    """
    Wrap the data loader to visualize its output via TensorBoardX at given frequency.
    """

    def __init__(self, cfg, tbx_writer, data_loader):
        self.tbx_writer = tbx_writer
        self.data_loader = data_loader
        self._visualizer = VisualizerWrapper(cfg)

        self.log_frequency = cfg.TENSORBOARD.TRAIN_LOADER_VIS_WRITE_PERIOD
        self.log_limit = cfg.TENSORBOARD.TRAIN_LOADER_VIS_MAX_IMAGES
        assert self.log_frequency >= 0
        assert self.log_limit >= 0
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

        length = min(len(data), self._remaining)
        data = data[:length]
        self._remaining -= length

        for i, per_image in enumerate(data):
            vis_image = self._visualizer.visualize_train_input(per_image)
            tag = "train_loader_batch_{}/".format(storage.iter)
            if "dataset_name" in per_image:
                tag += per_image["dataset_name"] + "/"
            if "file_name" in per_image:
                tag += "img_{}/{}".format(i, per_image["file_name"])
            self.tbx_writer._writer.add_image(
                tag=tag,
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
        self, cfg, tbx_writer, dataset_mapper, dataset_name, train_iter=None, tag_postfix=None
    ):
        self.tbx_writer = tbx_writer
        self.dataset_mapper = dataset_mapper
        self.dataset_name = dataset_name
        self._visualizer = VisualizerWrapper(cfg)
        self.train_iter = train_iter or VisualizationEvaluator._counter
        self.tag_postfix = tag_postfix or ""

        self.log_limit = max(cfg.TENSORBOARD.TEST_VIS_MAX_IMAGES, 0)
        if self.log_limit > 0:
            self._metadata = MetadataCatalog.get(dataset_name)
            # NOTE: Since there's no GT from test loader, we need to get GT from
            # the dataset_dict, this assumes the test data loader uses the item from
            # dataset_dict in the default way.
            self._dataset_dict = DatasetCatalog.get(dataset_name)
            self._file_name_to_dataset_dict = {
                dic["file_name"]: dic for dic in self._dataset_dict
            }

        VisualizationEvaluator._counter += 1
        self.reset()

    def reset(self):
        self._iter = 0
        self._log_remaining = self.log_limit

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            if self._log_remaining <= 0:
                return

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
