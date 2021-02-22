#!/usr/bin/env python3

from typing import Optional

import chamnet.search_space as ss
from mobile_cv.common.misc.registry import Registry


D2GO_SEARCH_SPACE_REGISTRY = Registry("D2GO_SEARCH_SPACE")
D2GO_PREDEFINED_ARCH_REGISTRY = Registry("D2GO_PREDEFINED_ARCH")
D2GO_SEARCH_SPACE_FILTER_REGISTRY = Registry("D2GO_SEARCH_SPACE_FILTER")


def build_search_space(
    search_space_name: str,
    predefined_arch_name: Optional[str] = None,
    filter_name: Optional[str] = None,
):
    """Create search space using registered values

    The search space consists of a dictionary which defines architectures with
    searchrange objects, list of predefined_arch that will be trained along
    with any samples and a filter that removes search space values

    Args:
        search_space_name (str): name of the registered search space to use
        predefined_arch_name (Optional[str]): name of registered predefined archs
        filter_name (Optional[str]): name of a search space filter
    """
    search_space_dict = D2GO_SEARCH_SPACE_REGISTRY.get(search_space_name)

    predefined_arch_list = None
    if predefined_arch_name is not None:
        predefined_arch_list = D2GO_PREDEFINED_ARCH_REGISTRY.get(predefined_arch_name)

    filt = None
    if filter_name is not None:
        filt = D2GO_SEARCH_SPACE_FILTER_REGISTRY.get(filter_name)

    return ss.SearchSpace(search_space_dict, predefined_arch_list, filt)
