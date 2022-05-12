from mobile_cv.common.misc.registry import Registry

QCONFIG_CREATOR_REGISTRY = Registry("QCONFIG_CREATOR_REGISTRY")


def set_backend_and_create_qconfig(cfg, *, is_train):
    """
    Recommended function to create qconfig given D2Go's quantization config.
    """

    # In case we need different implmentation, we can add a new key called
    # QUANTIZATION.QCONFIG_CREATOR with "smart" as default value, and use this key
    # to toggle between registries.
    return QCONFIG_CREATOR_REGISTRY.get("smart")(cfg, is_train=is_train)
