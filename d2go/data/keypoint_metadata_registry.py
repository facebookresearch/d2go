#!/usr/bin/env python3

from typing import List, NamedTuple, Tuple

from detectron2.utils.registry import Registry


KEYPOINT_METADATA_REGISTRY = Registry("KEYPOINT_METADATA")
KEYPOINT_METADATA_REGISTRY.__doc__ = "Registry keypoint metadata definitions"


class KeypointMetadata(NamedTuple):
    names: List[str]
    flip_map: List[Tuple[str, str]]
    connection_rules: List[Tuple[str, str, Tuple[int, int, int]]]

    def to_dict(self):
        return {
            "keypoint_names": self.names,
            "keypoint_flip_map": self.flip_map,
            "keypoint_connection_rules": self.connection_rules,
        }


def get_keypoint_metadata(name):
    return KEYPOINT_METADATA_REGISTRY.get(name)().to_dict()
