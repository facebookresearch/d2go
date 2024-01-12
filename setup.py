#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import glob
import os
import shutil
from os import path
from typing import List

from setuptools import find_packages, setup

cwd = os.path.dirname(os.path.abspath(__file__))

version = "0.0.1"
try:
    if not os.getenv("RELEASE"):
        from datetime import date

        today = date.today()
        day = today.strftime("b%Y%m%d")
        version += day
except Exception:
    pass

requirements = [
    "importlib",
    "numpy",
    "Pillow",
    "mock",
    "torch",
    "pytorch-lightning==1.8.6",
    "opencv-python",
    "parameterized",
    # Downgrade the protobuf package to 3.20.x or lower, related:
    # https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
    # https://github.com/protocolbuffers/protobuf/issues/10051
    "protobuf==3.20.2",
    "torchtnt",
]


def d2go_gather_files(dst_module, file_path, extension="*") -> List[str]:
    """
    Return a list of files to include in d2go submodule. Copy over the corresponding files.
    """
    # Use absolute paths while symlinking.
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), file_path)
    destination = path.join(path.dirname(path.realpath(__file__)), "d2go", dst_module)
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if path.exists(source_configs_dir):
        if path.islink(destination):
            os.unlink(destination)
        elif path.isdir(destination):
            shutil.rmtree(destination)

    if not path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob(os.path.join(file_path + extension), recursive=True)
    return config_paths


if __name__ == "__main__":
    setup(
        name="d2go",
        version=version,
        author="Mobile Vision",
        url="https://github.com/facebookresearch/d2go",
        description="D2Go",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        license="Apache-2.0",
        install_requires=requirements,
        packages=find_packages(exclude=["tools", "tests"]),
        package_data={
            "d2go": [
                "LICENSE",
            ],
            "d2go.configs": d2go_gather_files("configs", "configs", "**/*.yaml"),
            "d2go.tools": d2go_gather_files("tools", "tools", "**/*.py"),
            "d2go.tests": d2go_gather_files("tests", "tests", "**/*helper.py"),
        },
        entry_points={
            "console_scripts": [
                "d2go.exporter = d2go.tools.exporter:cli",
                "d2go.train_net = d2go.tools.train_net:cli",
            ]
        },
    )
