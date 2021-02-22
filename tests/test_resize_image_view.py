#!/usr/bin/env python3

import random
from typing import Tuple

import numpy as np
from d2go.runner.argos.argos_datatypes import (
    CameraModel,
    ImageLayer,
    ImageLayerType,
    ImageView,
)
from d2go.runner.argos.augmentor.resize_image_view import resize_image_view
from d2go.runner.argos.geometry.cameras import ProjectionLinearCamera
from unittest import TestCase


class ResizeImageViewTestCase(TestCase):
    def _create_depth_image_view(
        self, image_size_hw: Tuple[int, int], add_camera: bool, add_c_axis: bool
    ) -> ImageView:

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

        if add_c_axis:
            mask = mask[..., np.newaxis]
            depth = depth[..., np.newaxis]

        return ImageView(
            name="test_depth_view",
            datum=ImageLayer(type=ImageLayerType.image, file_name=None, data=image),
            gt=ImageLayer(type=ImageLayerType.depth, file_name=None, data=depth),
            mask=ImageLayer(type=ImageLayerType.mask, file_name=None, data=mask),
            camera=camera if add_camera else None,
            T_view_world=None,
        )

    def test_resize_image_works(self) -> None:

        for add_camera in [True, False]:
            for add_c_axis in [True, False]:
                input_image_size_hw = (random.randint(1, 11), random.randint(1, 11))

                output_image_size_hw = (random.randint(1, 11), random.randint(1, 11))

                resize_view = resize_image_view(
                    view=self._create_depth_image_view(
                        image_size_hw=input_image_size_hw,
                        add_camera=add_camera,
                        add_c_axis=add_c_axis,
                    ),
                    output_image_size_hw=output_image_size_hw,
                )

                self.assertEqual(resize_view.datum.data.shape[:2], output_image_size_hw)
                self.assertEqual(resize_view.gt.data.shape[:2], output_image_size_hw)
                if add_c_axis:
                    self.assertEqual(resize_view.gt.data.shape[-1], 1)
                    self.assertEqual(resize_view.mask.data.shape[-1], 1)

                self.assertEqual(resize_view.mask.data.shape[:2], output_image_size_hw)

                if add_camera:
                    self.assertIsNotNone(resize_view.camera)
                else:
                    self.assertIsNone(resize_view.camera)

    # TODO akashbapat: Need to add data test for scaling of disparity and flow.
