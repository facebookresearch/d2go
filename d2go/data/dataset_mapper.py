#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import logging

# TODO: Deprecate this file in favor of the module detectron2go.data.dataset_mappers
from d2go.data.dataset_mappers import D2GoDatasetMapper  # noqa
from PIL import Image


logger = logging.getLogger(__name__)


_IMAGE_LOADER_REGISTRY = {}


def register_uri_image_loader(scheme, loader):
    """
    Image can be represented as "scheme://path", image will be retrived by calling
        Image.open(loader(path)).
    """
    logger.info(
        "Register image loader for scheme: {} with loader: {}".format(scheme, loader)
    )
    _IMAGE_LOADER_REGISTRY[scheme] = loader


# TODO (T62922909): remove UniversalResourceLoader and use PathManager
class UniversalResourceLoader(object):
    def __init__(self):
        self._image_loader_func_map = copy.deepcopy(_IMAGE_LOADER_REGISTRY)

    @staticmethod
    def parse_path(uri):
        SCHEME_SEPARATOR = "://"
        if uri.count(SCHEME_SEPARATOR) < 1:
            # this should be a normal file name, use full string as path
            return "file", uri

        scheme, path = uri.split(SCHEME_SEPARATOR, maxsplit=1)
        return scheme, path

    def get_file(self, uri):
        scheme, path = self.parse_path(uri)
        if scheme not in self._image_loader_func_map:
            raise RuntimeError(
                "No loader for scheme {} in UniversalResourceLoader for uri: {}".format(
                    scheme, uri
                )
            )

        loader = self._image_loader_func_map[scheme]
        return loader(path)

    def support(self, dataset_dict):
        uri = dataset_dict["file_name"]
        scheme, _ = self.parse_path(uri)
        return scheme in self._image_loader_func_map

    def __call__(self, dataset_dict):
        uri = dataset_dict["file_name"]
        fp = self.get_file(uri)
        return Image.open(fp)

    def __repr__(self):
        return "UniversalResourceLoader(schemes={})".format(
            list(self._image_loader_func_map.keys())
        )
