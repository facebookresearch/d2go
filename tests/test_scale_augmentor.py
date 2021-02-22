#!/usr/bin/env python3


from typing import List, Tuple

import numpy as np
from d2go.runner.argos.argos_datatypes import (
    CameraModel,
    ImageLayer,
    ImageLayerType,
    ImageView,
)
from d2go.runner.argos.augmentor.scale_augmentator import ScaleAugmentor
from d2go.runner.argos.geometry.cameras import ProjectionLinearCamera
from unittest import TestCase


class ScaleAugmentorTestCase(TestCase):
    def _create_depth_image_view(self, image_size_hw: Tuple[int, int]) -> ImageView:

        camera = ProjectionLinearCamera(
            camera_model=CameraModel(
                params=np.array([12, 13, image_size_hw[1] / 2, image_size_hw[0] / 2])
            )
        )

        image = (
            np.random.random((image_size_hw[0], image_size_hw[1], 3)) * 255
        ).astype(np.uint8)
        depth = np.random.random((image_size_hw[0], image_size_hw[1]))
        mask = (depth > 0.5).astype(np.bool)

        mask = mask[..., np.newaxis]
        depth = depth[..., np.newaxis]

        return ImageView(
            name="test_depth_view",
            datum=ImageLayer(type=ImageLayerType.image, file_name=None, data=image),
            gt=ImageLayer(type=ImageLayerType.depth, file_name=None, data=depth),
            mask=ImageLayer(type=ImageLayerType.mask, file_name=None, data=mask),
            camera=camera,
            T_view_world=None,
        )

    def test_scaled_output_is_within_bounds(self) -> None:
        image_size_hw = (5, 7)
        views: List[ImageView] = []
        for _ in range(2):
            view = self._create_depth_image_view(image_size_hw=image_size_hw)
            views.append(view)

        scale_min_max = (0.9, 4)
        augmentor = ScaleAugmentor(scale_min_max=scale_min_max)
        augmented_views = augmentor(views)

        for view in augmented_views:
            for dim in range(0, 2):
                self.assertGreaterEqual(
                    view.datum.data.shape[dim],
                    round(scale_min_max[0] * image_size_hw[dim]),
                )
                self.assertLessEqual(
                    view.datum.data.shape[dim],
                    round(scale_min_max[1] * image_size_hw[dim]),
                )
                self.assertEqual(view.gt.data.shape[dim], view.datum.data.shape[dim])
                self.assertEqual(view.mask.data.shape[dim], view.datum.data.shape[dim])

    def test_augmentor_fails_with_size_zero_scaling(self) -> None:
        image_size_hw = (5, 7)
        views: List[ImageView] = []
        for _ in range(2):
            view = self._create_depth_image_view(image_size_hw=image_size_hw)
            views.append(view)

        # THe min max scaling factors are too aggressive.
        scale_min_max = (0.05, 0.1)
        augmentor = ScaleAugmentor(scale_min_max=scale_min_max)
        self.assertRaises(AssertionError, augmentor.__call__, views)
