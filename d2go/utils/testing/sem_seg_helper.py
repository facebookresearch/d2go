#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch


def generate_test_input(height, width, is_train, num_classes, super_classes=None):
    random_image = torch.rand(3, height, width).to(torch.float32)
    ret = {"image": random_image}
    if is_train:
        mask_size = (
            (height, width)
            if super_classes is None
            else (len(super_classes), height, width)
        )
        random_mask = torch.randint(low=0, high=num_classes, size=mask_size).to(
            torch.int64
        )
        ret["sem_seg"] = random_mask
    return ret


def validate_test_output(output, height, width, num_classes, super_classes=None):
    sem_seg_per_image = output["sem_seg"]

    if super_classes is None:  # None MCS case
        detect_c_out, detect_h_out, detect_w_out = sem_seg_per_image.size()
        assert detect_c_out == num_classes, detect_c_out
        assert detect_h_out == height, (detect_h_out, height)
        assert detect_w_out == width, (detect_w_out, width)
    else:  # MCS case
        assert isinstance(sem_seg_per_image, dict)
        assert all(k in super_classes for k in sem_seg_per_image), (
            sem_seg_per_image.keys(),
            super_classes,
        )
        for class_name, mask in sem_seg_per_image.items():
            assert isinstance(class_name, str)
            detect_c_out, detect_h_out, detect_w_out = mask.size()
            assert detect_c_out == num_classes, detect_c_out
            assert detect_h_out == height, (detect_h_out, height)
            assert detect_w_out == width, (detect_w_out, width)
