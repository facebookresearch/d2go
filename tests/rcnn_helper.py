#!/usr/bin/env python3

from typing import Optional

import d2go.data.transforms.box_utils as bu
import torch
from detectron2.structures import (
    Boxes,
    Instances,
)


def _get_image_with_box(image_size, boxes: Optional[Boxes] = None):
    """ Draw boxes on the image, one box per channel, use values 10, 20, ... """
    ret = torch.zeros((3, image_size[0], image_size[1]))
    if boxes is None:
        return ret
    assert len(boxes) <= ret.shape[0]
    for idx, box in enumerate(boxes):
        x0, y0, x1, y1 = box.int().tolist()
        ret[idx, y0:y1, x0:x1] = (idx + 1) * 10
    return ret


def _get_boxes_from_image(image, scale_xy=None):
    """Extract boxes from image created by `_get_image_with_box()`"""
    cur_img_int = ((image / 10.0 + 0.5).int().float() * 10.0).int()
    values = torch.unique(cur_img_int)
    gt_values = [x * 10 for x in range(len(values))]
    assert set(values.tolist()) == set(gt_values)
    boxes = []
    for idx in range(cur_img_int.shape[0]):
        val = torch.unique(cur_img_int[idx]).tolist()
        val = max(val)
        if val == 0:
            continue
        # mask = (cur_img_int[idx, :, :] == val).int()
        mask = (cur_img_int[idx, :, :] > 0).int()
        box_xywh = bu.get_box_from_mask(mask.numpy())
        boxes.append(bu.to_boxes_from_xywh(box_xywh))
    ret = Boxes.cat(boxes)
    if scale_xy is not None:
        ret.scale(*scale_xy)
    return ret


def get_batched_inputs(
    num_images,
    image_size=(1920, 1080),
    resize_size=(398, 224),
    boxes: Optional[Boxes] = None,
):
    """Get batched inputs in the format from d2/d2go data mapper
    Draw the boxes on the images if `boxes` is not None
    """
    ret = []

    for idx in range(num_images):
        cur = {
            "file_name": f"img_{idx}.jpg",
            "image_id": idx,
            "dataset_name": "test_dataset",
            "height": image_size[0],
            "width": image_size[1],
            "image": _get_image_with_box(resize_size, boxes),
        }
        ret.append(cur)

    return ret


def _get_keypoints_from_boxes(boxes: Boxes, num_keypoints: int):
    """ Use box center as keypoints """
    centers = boxes.get_centers()
    kpts = torch.cat((centers, torch.ones(centers.shape[0], 1)), dim=1)
    kpts = kpts.repeat(1, num_keypoints).reshape(len(boxes), num_keypoints, 3)
    return kpts


def _get_scale_xy(output_size_hw, instance_size_hw):
    return (
        output_size_hw[1] / instance_size_hw[1],
        output_size_hw[0] / instance_size_hw[0],
    )


def get_detected_instances_from_image(batched_inputs, scale_xy=None):
    """Get detected instances from batched_inputs, the results are in the same
    format as GeneralizedRCNN.inference()
    The images in the batched_inputs are created by `get_batched_inputs()` with
    `boxes` provided.
    """
    ret = []
    for item in batched_inputs:
        cur_img = item["image"]
        img_hw = cur_img.shape[1:]
        boxes = _get_boxes_from_image(cur_img, scale_xy=scale_xy)
        num_boxes = len(boxes)
        fields = {
            "pred_boxes": boxes,
            "scores": torch.Tensor([1.0] * num_boxes),
            "pred_classes": torch.Tensor([0] * num_boxes).int(),
            "pred_keypoints": _get_keypoints_from_boxes(boxes, 21),
            "pred_keypoint_heatmaps": torch.ones([num_boxes, 21, 24, 24]),
        }
        ins = Instances(img_hw, **fields)
        ret.append(ins)
    return ret


def get_detected_instances(num_images, num_instances, resize_size=(392, 224)):
    """Create an detected instances for unit test"""
    assert num_instances in [1, 2]

    ret = []
    for _idx in range(num_images):
        fields = {
            "pred_boxes": Boxes(torch.Tensor([[50, 40, 100, 80], [150, 60, 200, 120]])),
            "scores": torch.Tensor([1.0, 1.0]),
            "pred_classes": torch.Tensor([0, 0]).int(),
            "pred_keypoints": torch.Tensor(
                [70, 60, 1.5] * 21 + [180, 100, 2.0] * 21
            ).reshape(2, 21, 3),
            "pred_keypoint_heatmaps": torch.ones([2, 21, 24, 24]),
        }
        ins = Instances(resize_size, **fields)[:num_instances]
        ret.append(ins)

    return ret


class MockRCNNInference(object):
    """Use to mock the GeneralizedRCNN.inference()"""

    def __init__(self, image_size, resize_size):
        self.image_size = image_size
        self.resize_size = resize_size

    @property
    def device(self):
        return torch.device("cpu")

    def __call__(
        self,
        batched_inputs,
        detected_instances=None,
        do_postprocess: bool = True,
    ):
        return self.inference(
            batched_inputs,
            detected_instances,
            do_postprocess,
        )

    def inference(
        self,
        batched_inputs,
        detected_instances=None,
        do_postprocess: bool = True,
    ):
        scale_xy = (
            _get_scale_xy(self.image_size, self.resize_size) if do_postprocess else None
        )
        results = get_detected_instances_from_image(batched_inputs, scale_xy=scale_xy)
        # when do_postprocess is True, the result instances is stored inside a dict
        if do_postprocess:
            results = [{"instances": r} for r in results]

        return results
