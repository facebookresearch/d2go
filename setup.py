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


def get_model_zoo_configs() -> List[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    d2go/model_zoo.
    """

    # Use absolute paths while symlinking.
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
    destination = path.join(
        path.dirname(path.realpath(__file__)), "d2go", "model_zoo", "configs"
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

    config_paths = glob.glob("configs/**/*.yaml", recursive=True)
    return config_paths

def get_test_helper() -> List[str]:
    """
    Return a list of helper file to include in package for tests. Copy over these file inside
    d2go/tests.
    """

    # Use absolute paths while symlinking.
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "tests")
    destination = path.join(
        path.dirname(path.realpath(__file__)), "d2go", "tests"
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

    config_paths = glob.glob("tests/**/*helper.py", recursive=True)
    return config_paths

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
            "d2go.tests": get_test_helper(),
        },
    )

