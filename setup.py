#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import os
import subprocess
import glob
import shutil
from os import path
from typing import List

from setuptools import setup, find_packages

cwd = os.path.dirname(os.path.abspath(__file__))

version = '0.0.1'
try:
    if not os.getenv('RELEASE'):
        from datetime import date
        today = date.today()
        day = today.strftime("b%Y%m%d")
        version += day
except Exception:
    pass

requirements = [
    'importlib',
    'numpy',
    'Pillow',
    'mock',
    'torch',
    'pytorch_lightning',
    'opencv-python',
]

def d2go_gather_files(dst_module, file_path, extension="*") -> List[str]:
    """
    Return a list of files to include in d2go submodule. Copy over the corresponding files.
    """
    # Use absolute paths while symlinking.
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), file_path)
    destination = path.join(
        path.dirname(path.realpath(__file__)), "d2go", dst_module
    )
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

def get_model_zoo_configs() -> List[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    d2go/model_zoo.
    """
    return d2go_gather_files(os.path.join("model_zoo", "configs"), "configs", "**/*.yaml")

if __name__ == '__main__':
    setup(
        name="d2go",
        version=version,
        author="Mobile Vision",
        url="https://github.com/facebookresearch/d2go",
        description="D2Go",
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        license='Apache-2.0',
        install_requires=requirements,
        packages=find_packages(exclude=["tools", "tests"]),
        package_data={'resnest': [
                'LICENSE',
            ],
            "d2go.model_zoo": get_model_zoo_configs(),
            "d2go.tests": d2go_gather_files("tests", "tests", "**/*helper.py"),
            "d2go.tools": d2go_gather_files("tools", "tools", "**/*.py"),
        },
        entry_points={
            'console_scripts': [
                'd2go.exporter = d2go.tools.exporter:cli',
                'd2go.train_net = d2go.tools.train_net:cli',
            ]
        },
    )

