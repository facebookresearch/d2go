#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pkg_resources


def get_resource_from_package(package: str, relative_path: str) -> str:
    return pkg_resources.resource_filename(package, relative_path)
