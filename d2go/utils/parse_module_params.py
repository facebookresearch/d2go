#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


def iterate_module_named_parameters(model, check_requires_grad=True):
    """Iterate over all parameters for the model"""
    memo = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if check_requires_grad and not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            yield module_name, module, module_param_name, value
