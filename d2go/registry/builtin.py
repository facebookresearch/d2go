#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from mobile_cv.common.misc.registry import Registry

"""
This file contains all D2Go's builtin registries with global scope.

- These registries can be treated as "static". There'll be a bootstrap process happens
    at the beginning of the program to make it works like the registrations happen
    at compile time (like C++). In another word, the objects are guaranteed to be
    registered to those builtin registries without user importing their code.

- Since the namespace is global, the registered name has to be unique across all projects.
"""

DEMO_REGISTRY = Registry("DEMO")

# Registry for config updater
CONFIG_UPDATER_REGISTRY = Registry("CONFIG_UPDATER")
