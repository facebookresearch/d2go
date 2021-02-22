import re
import logging

logger = logging.getLogger(__name__)


def add_model_freezing_configs(_C):
    _C.MODEL.FROZEN_LAYER_REG_EXP = []


def set_requires_grad(model, reg_exps, value):
    total_num_parameters = 0
    unmatched_parameters = []
    unmatched_parameter_names = []
    matched_parameters = []
    matched_parameter_names = []
    for name, parameter in model.named_parameters():
        total_num_parameters += 1
        matched = False
        for frozen_layers_regex in reg_exps:
            if re.match(frozen_layers_regex, name):
                matched = True
                parameter.requires_grad = value
                matched_parameter_names.append(name)
                matched_parameters.append(parameter)
                break
        if not matched:
            unmatched_parameter_names.append(name)
            unmatched_parameters.append(parameter)
    logger.info("Matched layers (require_grad={}): {}".format(
        value, matched_parameter_names))
    logger.info("Unmatched layers: {}".format(unmatched_parameter_names))
    return matched_parameter_names, unmatched_parameter_names
