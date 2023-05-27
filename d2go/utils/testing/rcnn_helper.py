#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import shutil
import tempfile
import unittest
from typing import Optional

import d2go.data.transforms.box_utils as bu
import torch
from d2go.export.exporter import convert_and_export_predictor
from d2go.runner.default_runner import GeneralizedRCNNRunner
from d2go.utils.testing.data_loader_helper import (
    create_detection_data_loader_on_toy_dataset,
)
from detectron2.structures import Boxes, Instances
from mobile_cv.predictor.api import create_predictor
from parameterized import parameterized


def _get_image_with_box(image_size, boxes: Optional[Boxes] = None):
    """Draw boxes on the image, one box per channel, use values 10, 20, ..."""
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
    """Use box center as keypoints"""
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


def _validate_outputs(inputs, outputs):
    assert len(inputs) == len(outputs)
    # TODO: figure out how to validate outputs


def get_quick_test_config_opts(
    fixed_single_proposals=True,
    small_pooler_resolution=True,
    small_resize_resolution=True,
):
    ret = []
    if fixed_single_proposals:
        epsilon = 1e-4
        ret.extend(
            [
                "MODEL.RPN.POST_NMS_TOPK_TEST",
                1,
                "TEST.DETECTIONS_PER_IMAGE",
                1,
                "MODEL.PROPOSAL_GENERATOR.MIN_SIZE",
                0,
                "MODEL.RPN.NMS_THRESH",
                1.0 + epsilon,
                "MODEL.ROI_HEADS.NMS_THRESH_TEST",
                1.0 + epsilon,
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                0.0 - epsilon,
            ]
        )
    if small_pooler_resolution:
        ret.extend(
            [
                "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION",
                1,
                "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION",
                1,
                "MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION",
                1,
            ]
        )
    if small_resize_resolution:
        ret.extend(
            [
                "INPUT.MIN_SIZE_TRAIN",
                (8,),
                "INPUT.MAX_SIZE_TRAIN",
                9,
                "INPUT.MIN_SIZE_TEST",
                10,
                "INPUT.MAX_SIZE_TEST",
                11,
            ]
        )
    return [str(x) for x in ret]


def get_export_test_name(testcase_func, param_num, param):
    predictor_type, compare_match = param.args
    assert isinstance(predictor_type, str)
    assert isinstance(compare_match, bool)

    return "{}_{}".format(
        testcase_func.__name__, parameterized.to_safe_name(predictor_type)
    )


class RCNNBaseTestCases:
    @staticmethod
    def expand_parameterized_test_export(*args, **kwargs):
        if "name_func" not in kwargs:
            kwargs["name_func"] = get_export_test_name
        return parameterized.expand(*args, **kwargs)

    class TemplateTestCase(unittest.TestCase):  # TODO: maybe subclass from TestMetaArch
        def setUp(self):
            self.setup_test_dir()
            assert hasattr(self, "test_dir")

            self.setup_custom_test()
            assert hasattr(self, "runner")
            assert hasattr(self, "cfg")
            self.force_apply_overwrite_opts()

            self.test_model = self.runner.build_model(self.cfg, eval_only=True)

        def setup_test_dir(self):
            self.test_dir = tempfile.mkdtemp(prefix="test_export_")
            self.addCleanup(shutil.rmtree, self.test_dir)

        def _get_test_image_sizes_default(self, is_train):
            # model should work for any size, so don't alway use power of 2 or multiple
            # of size_divisibility for testing.
            side_length = max(self.test_model.backbone.size_divisibility, 10)
            # make it non-square to cover error caused by messing up width & height
            h, w = side_length, side_length * 2
            return h, w

        def _get_test_image_size_no_resize(self, is_train):
            # use cfg.INPUT to make sure data loader doesn't resize the image
            if is_train:
                assert len(self.cfg.INPUT.MAX_SIZE_TRAIN) == 1
                h = self.cfg.INPUT.MIN_SIZE_TRAIN[0]
                w = self.cfg.INPUT.MAX_SIZE_TRAIN
            else:
                h = self.cfg.INPUT.MIN_SIZE_TEST
                w = self.cfg.INPUT.MAX_SIZE_TEST
            return h, w

        def _get_test_image_sizes(self, is_train):
            """override this method to use other image size strategy"""
            return self._get_test_image_sizes_default(is_train)

        def setup_custom_test(self):
            """
            Override this when using different runner, using different base config file,
            or setting specific config for certain test.
            """
            self.runner = GeneralizedRCNNRunner()
            self.cfg = self.runner.get_default_cfg()
            # subclass can call: self.cfg.merge_from_file(...)

        def force_apply_overwrite_opts(self):
            """
            Recommend only overriding this for a group of tests, while indivisual test
            should have its own `setup_custom_test`.
            """
            # update config to make the model run fast
            self.cfg.merge_from_list(get_quick_test_config_opts())
            # forcing test on CPU
            self.cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

        @contextlib.contextmanager
        def _create_data_loader(self, is_train):
            """
            Creating the data loader used for the test case. Note that it's better
            to use "fake" data for quick test and isolating I/O.
            """
            image_height, image_width = self._get_test_image_sizes(is_train=False)
            with create_detection_data_loader_on_toy_dataset(
                self.cfg,
                image_height,
                image_width,
                is_train=is_train,
                runner=self.runner,
            ) as data_loader:
                yield data_loader

        def _test_export(self, predictor_type, compare_match=True):
            with self._create_data_loader(is_train=False) as data_loader:
                inputs = next(iter(data_loader))

                # TODO: the export may change model it self, need to fix this
                model_to_export = copy.deepcopy(self.test_model)
                predictor_path = convert_and_export_predictor(
                    self.cfg,
                    model_to_export,
                    predictor_type,
                    self.test_dir,
                    data_loader,
                )

                predictor = create_predictor(predictor_path)
                predictor_outputs = predictor(inputs)
                _validate_outputs(inputs, predictor_outputs)

                if compare_match:
                    with torch.no_grad():
                        pytorch_outputs = self.test_model(inputs)

                    from detectron2.utils.testing import assert_instances_allclose

                    assert_instances_allclose(
                        predictor_outputs[0]["instances"],
                        pytorch_outputs[0]["instances"],
                        size_as_tensor=True,
                    )

            return predictor_path

        # TODO: add test_train

        def _test_inference(self):
            with self._create_data_loader(is_train=False) as data_loader:
                inputs = next(iter(data_loader))

            with torch.no_grad():
                outputs = self.test_model(inputs)
            _validate_outputs(inputs, outputs)
