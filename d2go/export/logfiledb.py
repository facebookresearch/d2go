#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import numpy as np

from mobile_cv.torch.utils_caffe2.ws_utils import ScopedWS


logger = logging.getLogger(__name__)


# NOTE: specific export_to_db for (data, im_info) dual inputs.
# modified from mobile-vision/common/utils/model_utils.py
def export_to_db(net, params, inputs, outputs, out_file, net_type=None, shapes=None):
    # NOTE: special handling for im_info: by default the "predict_init_net"
    # will zero_fill inputs/outputs (https://fburl.com/diffusion/nvksomrt),
    # however the actual value of "im_info" also matters, so we need use
    # extra_init_net to handle this.
    import numpy as np
    from caffe2.python import core

    assert len(inputs) == 2
    data_name, im_info_name = inputs
    data_shape = shapes[data_name]  # assume NCHW
    extra_init_net = core.Net("extra_init_net")
    im_info = np.array(
        [[data_shape[2], data_shape[3], 1.0] for _ in range(data_shape[0])],
        dtype=np.float32,
    )
    extra_init_net.GivenTensorFill(
        [], im_info_name, shape=shapes[im_info_name], values=im_info
    )

    from caffe2.caffe2.fb.predictor import predictor_exporter  # NOTE: slow import

    predictor_export_meta = predictor_exporter.PredictorExportMeta(
        predict_net=net,
        parameters=params,
        inputs=inputs,
        outputs=outputs,
        net_type=net_type,
        shapes=shapes,
        extra_init_net=extra_init_net,
    )

    logger.info("Writing logdb {} ...".format(out_file))
    predictor_exporter.save_to_db(
        db_type="log_file_db",
        db_destination=out_file,
        predictor_export_meta=predictor_export_meta,
    )


def export_to_logfiledb(predict_net, init_net, outfile, ws_blobs):
    logger.info("Exporting Caffe2 model to {}".format(outfile))

    shapes = {
        b: data.shape if isinstance(data, np.ndarray)
        # proivde a dummpy shape if it could not be inferred
        else [1]
        for b, data in ws_blobs.items()
    }

    with ScopedWS("__ws_tmp__", is_reset=True) as ws:
        ws.RunNetOnce(init_net)
        initialized_blobs = set(ws.Blobs())
        uninitialized = [
            inp for inp in predict_net.external_input if inp not in initialized_blobs
        ]

        params = list(initialized_blobs)
        output_names = list(predict_net.external_output)

        export_to_db(
            predict_net, params, uninitialized, output_names, outfile, shapes=shapes
        )
