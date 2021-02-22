#!/usr/bin/env python3

import logging
import torch
import os
from torch import nn
from typing import Dict, Tuple

from detectron2.export.api import Caffe2Model
from detectron2.export.caffe2_export import (
    export_caffe2_detection_model,
    run_and_save_graph,
)
from d2go.export.logfiledb import export_to_logfiledb

logger = logging.getLogger(__name__)


def export_caffe2(
    caffe2_compatible_model: nn.Module,
    tensor_inputs: Tuple[str, torch.Tensor],
    output_dir: str,
    save_pb: bool = True,
    save_logdb: bool = False,
) -> Tuple[Caffe2Model, Dict[str, str]]:
    predict_net, init_net = export_caffe2_detection_model(
        caffe2_compatible_model,
        # pyre-fixme[6]: Expected `List[torch.Tensor]` for 2nd param but got
        #  `Tuple[str, torch.Tensor]`.
        tensor_inputs,
    )

    caffe2_model = Caffe2Model(predict_net, init_net)

    caffe2_export_paths = {}
    if save_pb:
        caffe2_model.save_protobuf(output_dir)
        caffe2_export_paths.update({
            "predict_net_path": os.path.join(output_dir, "model.pb"),
            "init_net_path": os.path.join(output_dir, "model_init.pb"),
        })

    graph_save_path = os.path.join(output_dir, "model_def.svg")
    ws_blobs = run_and_save_graph(
        predict_net,
        init_net,
        tensor_inputs,
        graph_save_path=graph_save_path,
    )
    caffe2_export_paths.update({
        "model_def_path": graph_save_path,
    })

    if save_logdb:
        logfiledb_path = os.path.join(output_dir, "model.logfiledb")
        export_to_logfiledb(predict_net, init_net, logfiledb_path, ws_blobs)
        caffe2_export_paths.update({
            "logfiledb_path": logfiledb_path if save_logdb else None,
        })

    return caffe2_model, caffe2_export_paths
