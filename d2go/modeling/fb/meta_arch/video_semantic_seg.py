#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from detectron2.data import MetadataCatalog, detection_utils as utils
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.modeling.meta_arch.semantic_seg import build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.visualizer import Visualizer


## This meta_arch is similar to TemporalSemanticSegmentor in semantic_seg.py
## but trains and evaluates on video clips

@META_ARCH_REGISTRY.register()
class VideoSemanticSegmentor(nn.Module):
    """
    Detect-and-track version of D2's SemanticSegmentor
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

    @property
    def device(self):
        return self.pixel_mean.device

    def _preprocess_image(self, batched_inputs, size_divisibility):
        """
        Preprocess for a batch of inputs, where each input contains a sequence of
        images.

        Arguments:
            batched_inputs (List[Dict]): A minibatch of "videos", each video contains
                a list of images encoded in the "frames" key. Eg.
                [
                    {"frames": [clip_length, channel, height, width],
                    "height": height,
                    "width": width,
                    "sem_seg": []} # <- batch 1
                    # batch 2
                    ...
                ]
            size_divisibility: Size divisibility for all frames across all batches

        Returns:
            Dict[str, Tensor]: A dict of 4D tensor, the length of dict is equal to the
                number of frames, each tenosr is a batched across the minibatch.
        """

        images = [x["frames"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, size_divisibility)
        return images

    def _get_targets(self, batched_inputs, size_divisibility, ignore_value):
        """
        Similar to _preprocess_image
        """
        if "sem_seg" in batched_inputs[0]:
            targets_per_frame = {}
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, size_divisibility, ignore_value
            ).tensor
            targets_per_frame["video"] = targets
            return targets_per_frame
        else:
            return None

    def forward(self, batched_inputs):
        images = self._preprocess_image(
            batched_inputs, size_divisibility=self.backbone.size_divisibility
        )
        features = self.backbone(images.tensor)

        targets = self._get_targets(
            batched_inputs,
            size_divisibility=self.backbone.size_divisibility,
            ignore_value=self.sem_seg_head.ignore_value,
        )
        results, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses

        # NOTE: For single-image based SemanticSegmentor, "results" is a single
        # 4D tensor, here although using the same head registery, "results" should
        # be a dict of 4D tensors potentially representing results for different
        # frames.
        assert isinstance(results, dict)
        assert len(results) == len(images)
        assert all(isinstance(x, torch.Tensor) for x in results.values())

        processed_results = []
        for i in range(len(batched_inputs)):
            for _j in range(len(batched_inputs[i]["frames"])):
                processed_results.append({})

        for (name, results_per_name) in results.items():
            results_per_name = _convert_binary_logits_to_two_class_logits(
                results_per_name
            )
            index = 0
            image_size = images.tensor.shape[-2:]
            for batch_id in range(len(results_per_name)):
                height = batched_inputs[batch_id]["height"]
                width = batched_inputs[batch_id]["width"]
                for frame_i in range(len(results_per_name[batch_id])):
                    result = results_per_name[batch_id][frame_i]
                    r = sem_seg_postprocess(result, image_size, height, width)
                    processed_results[index][name] = {"sem_seg": r}
                    index += 1
        return processed_results

    @staticmethod
    def visualize_train_input(visualizer_wrapper, input_dict):
        per_image = input_dict
        cfg = visualizer_wrapper.cfg

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        scale = 2.0

        vis_images = []
        for i, frame in enumerate(per_image["frames"]):
            img = frame.permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            visualizer.draw_sem_seg(
                per_image["sem_seg"][i], area_threshold=0, alpha=0.5
            )
            visualizer.draw_text("video", (0, 0), horizontal_alignment="left")
            vis_images.append(visualizer.get_output().get_image())

        # putting all images side-by-side
        vis_img = np.concatenate(vis_images, axis=1)  # hwc
        return vis_img

    @staticmethod
    def visualize_test_output(
        visualizer_wrapper, dataset_name, dataset_mapper, input_dict, output_dict
    ):
        vis_images = []
        img = dataset_mapper._read_image(input_dict, "RGB")
        visualizer = Visualizer(img, metadata=MetadataCatalog.get(dataset_name))
        visualizer.draw_sem_seg(
            output_dict["sem_seg"].argmax(dim=0).to("cpu"), area_threshold=0, alpha=0.5
        )
        visualizer.draw_text("video", (0, 0), horizontal_alignment="left")
        vis_images.append(visualizer.get_output().get_image())

        # putting all images side-by-side
        vis_img = np.concatenate(vis_images, axis=1)  # hwc
        return vis_img


def _convert_binary_logits_to_two_class_logits(logits):
    assert len(logits.shape) >= 4  # (N1, N2, 1, H, W)
    if logits.shape[-3] == 2:
        return logits
    two_class_logits = torch.cat([-logits, logits], dim=-3)  # background  # foreground
    return two_class_logits
