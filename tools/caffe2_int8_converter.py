#!/usr/bin/env python3
"""
Binary to convert fp32 caffe2 model (outcome of caffe2_converter) to int8 caffe2 model.

This is the solution before native quantized model is ready, after that caffe2_convert
should be able to output int8 model directly.
"""

import logging
import numpy as np
import os
import torch
from typing import Any, Dict, List, Optional, Union

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.fb import int8_utils, hardcode_scale_zp
from fvcore.common.file_io import PathManager
from d2go.setup import (
    basic_argument_parser,
    prepare_for_launch,
    setup_after_launch,
)
from mobile_cv.common.misc.py import post_mortem_if_fail
from mobile_cv.torch.utils_caffe2.protobuf import (
    check_set_pb_arg,
    get_pb_arg_ints,
    get_pb_arg_vali,
    get_producer_map,
)
from mobile_cv.torch.utils_caffe2.state_transition import analyze_order, analyze_type
from mobile_cv.torch.utils_caffe2.vis import save_graph
from mobile_cv.torch.utils_caffe2.ws_utils import ScopedWS, fetch_any_blob
from mobile_cv.torch.utils_caffe2.graph_transform import (
    rename_op_output,
)

logger = logging.getLogger("d2go.tools.caffe2_int8_converter")


def _load_model(predict_net_path, init_net_path):
    predict_net = caffe2_pb2.NetDef()
    with PathManager.open(predict_net_path, "rb") as f:
        predict_net.ParseFromString(f.read())

    init_net = caffe2_pb2.NetDef()
    with PathManager.open(init_net_path, "rb") as f:
        init_net.ParseFromString(f.read())

    return predict_net, init_net


# TODO: it only works for int8 now
def fuse_conv_relu(proto):
    FUSIBLE_OP_PREV = {
        # "Relu": "Conv",
        "Int8Relu": "Int8Conv",
    }
    FUSED_TYPE = {
        # "Conv": "ConvRelu",
        "Int8Conv": "Int8ConvRelu",
    }
    ops = list(proto.op)
    new_ops = []
    for i, op in enumerate(ops):
        if op.type in FUSIBLE_OP_PREV:
            producer_ids = [id for id in range(i) if op.input[0] in ops[id].output]
            producers = [ops[i] for i in producer_ids]
            if len(producers) == 1 and producers[0].type == FUSIBLE_OP_PREV[op.type]:
                consumers = [x for x in ops[producer_ids[0] + 1:]
                    if producers[0].output[0] in x.input]
                # the prodcued blob should have only one consumer (relu)
                if len(consumers) > 1:
                    # the exception is that relu is inplace
                    assert consumers[0] == op and \
                        consumers[0].input[0] == consumers[0].output[0], \
                        "Consumers {}".format(consumers)
                else:
                    assert len(consumers) == 1, \
                        'Producers {}, Consumers {}'.format(producers, consumers)
                producers[0].output[0] = op.output[0]
                producers[0].type = FUSED_TYPE[producers[0].type]
                continue
            else:
                new_ops.append(op)
        else:
            new_ops.append(op)
    del proto.op[:]
    proto.op.extend(new_ops)


def _reset_engine(net):
    def remove_op_args_inplace(op, arg_names):
        for x in op.arg[:]:
            if x.name in arg_names:
                op.arg.remove(x)
        return op

    for op in net.Proto().op:
        if op.type in [
            "Conv", "ConvTranspose", "Int8Conv", "Int8ConvTranspose",
            "Int8ConvRelu",
        ]:
            op.engine = ""
            remove_op_args_inplace(op, ["exhaustive_search"])


def _get_1x1_pooling_ops(predict_net):
    ret = []
    for i, op in enumerate(predict_net.op):
        if op.type in ["MaxPool", "AveragePool"]:
            kernel = get_pb_arg_vali(op, "kernel", None)
            kernels = get_pb_arg_ints(op, "kernels", None)
            if kernel is None and kernels is None:
                continue
            elif kernel is not None and kernels is None:
                pooling_size = kernel * kernel
            elif kernel is None and kernels is not None:
                pooling_size = kernels[0] * kernels[1]
            else:
                raise ValueError("Can't determine pooling_size for op {}".format(op))

            if pooling_size <= 1:
                ret.append(i)
    return ret


def _get_if_skip_op(
    predict_net: caffe2_pb2.NetDef,
    fp32_blobs: Dict[str, Any],
    skip_first_conv: bool,
    ops_to_skip: Optional[List[str]],
    # TODO: maybe make the rool customizable
) -> List[bool]:
    """
    Return if skip each op in predict_net. One can modify this function to
    achieve other strategies.
    """

    # by op type
    OPS_TO_SKIP = [
        # "Softmax",  # Unsupported by DNNLOWP
        "Sigmoid",  # Results in accuracy drop even with 'followed_by'.
        # GenerateProposals is likely very sensitive to RPN scores.
    ]
    DNNLOWP_ONLY_OPS = [
        "BatchPermutation",  # int8_utils also can't do graph transfrom correctly for it
        "GroupNorm",
    ]
    OPS_TO_SKIP += DNNLOWP_ONLY_OPS
    # pyre-fixme[9]: ops_to_skip has type `Optional[List[str]]`; used as `Set[str]`.
    # pyre-fixme[9]: ops_to_skip has type `Optional[List[str]]`; used as `Set[str]`.
    ops_to_skip = set(ops_to_skip or []).union(OPS_TO_SKIP)
    # pyre-fixme[58]: `in` is not supported for right operand type
    #  `Optional[List[str]]`.
    # pyre-fixme[58]: `in` is not supported for right operand type
    #  `Optional[List[str]]`.
    if_skip_op = [op.type in ops_to_skip for op in predict_net.op]

    # by conv
    if skip_first_conv:
        for i, op in enumerate(predict_net.op):
            if op.type == "Conv":
                if_skip_op[i] = True
                break

    # NOTE: QNNPACK doesn't allow 1x1 pooling (although it still makes sense
    # sometimes), skip those ops automatically.
    for pooling_to_skip in _get_1x1_pooling_ops(predict_net):
        logger.warning("QNNPACK doesn't support 1x1 pooling, skip op:\n{}".format(
            predict_net.op[pooling_to_skip]
        ))
        if_skip_op[pooling_to_skip] = True

    # NOTE: int8_utils doesn't support FC with 4D input whose H,W != 1,1 yet.
    # Skip those ops automatically.
    for i, op in enumerate(predict_net.op):
        if op.type == "FC":
            assert op.input[0] in fp32_blobs, \
                "this requires fp32_blobs is versioned, otherwise we may need using ssa"
            input_shape = fp32_blobs[op.input[0]].shape
            if len(input_shape) == 4 and input_shape[-2:] != (1, 1):
                logger.warning(
                    "int8_utils doesn't support FC with 4D input whose H,W != 1,1"
                    " yet, got input shape: {}. skip op:\n{}".format(input_shape, op)
                )
                if_skip_op[i] = True

    return if_skip_op


def choose_quantization_params(net, init_net, quant_strat):
    if quant_strat == "L2_APPROX":
        quant_strat = hardcode_scale_zp.QuantizationStrategy.L2_MIN_QUANTIZATION_APPROX
    elif quant_strat == "MINMAX":
        quant_strat = hardcode_scale_zp.QuantizationStrategy.MIN_MAX_QUANTIZATION
    else:
        raise TypeError("Unknown quantization strategy: " + str(quant_strat))

    blobs_to_quant_params = hardcode_scale_zp.choose_quantization_params_for_net(
        init_net, net, quant_strat=quant_strat
    )

    return blobs_to_quant_params


def _run_net_tmp_ws(
    predict_net: Union[caffe2_pb2.NetDef, core.Net],
    init_net: Union[caffe2_pb2.NetDef, core.Net],
    test_inputs: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """ Run net in tmp workspace and return workspace blobs """
    with ScopedWS("__ws_tmp__", is_reset=True) as ws:
        try:
            ws.RunNetOnce(init_net)
            for name, blob in test_inputs.items():
                ws.FeedBlob(name, blob)
            ws.RunNetOnce(predict_net)
        except Exception as e:
            logger.warning("Run model failed: {}".format(e))
        return {b: fetch_any_blob(b) for b in ws.Blobs()}


def convert_detection_model_to_int8(
    predict_net: caffe2_pb2.NetDef,
    init_net: caffe2_pb2.NetDef,
    test_inputs: List[torch.Tensor],
    quant_strat: str,
    skip_first_conv: bool,
    ops_to_skip: Optional[List[str]],
    fuse_relu: bool,
    reset_engine: bool,
):
    # Remove non-initialized (eg. data, im_info) input from init_net
    # NOTE: model converted from detectron2go's caffe2_converter won't have this issue

    # turn things to core.Net
    predict_core_net = core.Net(predict_net)
    del predict_net
    init_core_net = core.Net(init_net)
    del init_net

    # pyre-fixme[16]: `List` has no attribute `items`.
    # pyre-fixme[16]: `List` has no attribute `items`.
    first_batch_inputs = {k: v[:1] for k, v in test_inputs.items()}

    # solving inplace blob issue
    int8_utils.add_version_to_inplace_blob(predict_core_net)

    logger.info("Running Net Once (single batch) to make sure fp32 model works ...")
    fp32_blobs = _run_net_tmp_ws(predict_core_net, init_core_net, first_batch_inputs)

    logger.info("Choosing quantization params ...")
    QUANT_STRAT_MAP = {
        "L2_APPROX": hardcode_scale_zp.QuantizationStrategy.L2_MIN_QUANTIZATION_APPROX,
        "MINMAX": hardcode_scale_zp.QuantizationStrategy.MIN_MAX_QUANTIZATION,
    }
    with ScopedWS("__caffe2_int8_converter__", is_reset=True) as ws:
        # NOTE: choose_quantization_params_for_net doesn't handle feeding
        # external inputs that are created by init_net (eg. data, im_info). Thus
        # we feed those inputs aforehand.
        for name, blob in test_inputs.items():
            ws.FeedBlob(name, blob)
        blobs_to_quant_params = hardcode_scale_zp.choose_quantization_params_for_net(
            init_core_net,
            predict_core_net,
            quant_strat=QUANT_STRAT_MAP[quant_strat]
        )
    blobs_to_quant_params_str = "".join(
        "{}: {}\n".format(k, blobs_to_quant_params[k])
        for k in sorted(blobs_to_quant_params.keys())
    )
    logger.info("blobs_to_quant_params:\n{}".format(blobs_to_quant_params_str))

    if_skip_op = _get_if_skip_op(
        predict_core_net.Proto(),
        fp32_blobs,
        skip_first_conv,
        ops_to_skip,
    )
    assert len(if_skip_op) == len(predict_core_net.Proto().op)
    if_skip_op_str = "".join(
        "(#{:>4}) {}\n".format(i, op.type)
        for i, op in enumerate(predict_core_net.Proto().op)
        if if_skip_op[i]
    )
    logger.info("Skip transform following operators: \n{}".format(if_skip_op_str))
    # NOTE: int8_net_transform can take lambda[op]->bool to specify if the op
    # does not need transform. To make it general, we encoded a field in the
    # op.arg and use it as identifier.
    for i, op in enumerate(predict_core_net.Proto().op):
        if if_skip_op[i]:
            check_set_pb_arg(op, "__skip_int8_transform__", "i", 1)

    # Convert RoIWarp/RoIAlign to NHWC since that's faster anyway.
    # This avoids unnecessary NCHW2NHWC and NHWC2NCHW conversions when
    # running int8_net_transform.
    # HACK: theoretically this is incorrect, hopefully after int8_net_transform
    # it'll in the correct order.
    from caffe2.python import utils
    # NOTE: This acutally causes crash for FPN models because roi_feat_fpn.1 will
    # still be in NCHW maybe because one of its consuer MaxPool is skipped.
    if False:
        logger.info("Change the order of RoIAlign to NHWC")
        for op in predict_core_net.Proto().op:
            if "RoIWarp" not in op.type and "RoIAlign" not in op.type:
                continue
            # Just change the arg. No need to change inputs/outputs
            # since int8_net_transform converts everything to NHWC.
            arg_list = [arg for arg in op.arg if arg.name != "order"]
            del op.arg[:]
            op.arg.extend(arg_list)
            op.arg.extend([utils.MakeArgument("order", "NHWC")])

    logger.info("Running int8_utils.int8_net_transform ...")
    # NOTE: int8_net_transform uses the blobs in the workspace, thus run it
    # under the same workspace that choose_quantization_params_for_net uses.
    with ScopedWS("__caffe2_int8_converter__", is_reset=False) as ws:
        q_init_net, q_net, qx_net = int8_utils.int8_net_transform(
            init_core_net,
            predict_core_net,
            blobs_to_quant_params=blobs_to_quant_params,
            extra_rule=lambda op: get_pb_arg_vali(op, "__skip_int8_transform__", 0),
        )

    # Run static order check, it will fail here before running the q_net
    # TODO: we may want to fix the order issue if we have the tool
    analyze_order(q_net.Proto())

    assert len(qx_net.Proto().op) == 0, NotImplementedError("Let me fix later")

    # int8_utils.remove_version_from_inplace_blob(net)

    if fuse_relu:
        fuse_conv_relu(q_net.Proto())

    if reset_engine:
        _reset_engine(q_net)

    logger.info("Running Int8 Net Once (single batch) to make sure it works")
    int8_blobs = _run_net_tmp_ws(q_net, q_init_net, first_batch_inputs)

    return q_net.Proto(), q_init_net.Proto(), fp32_blobs, int8_blobs


def int8_debug_info(net, init_net, FP32_BLOBS, int8_net, int8_init_net, INT8_BLOBS):
    def _int8_to_fp(ts):
        return (ts.data.astype(np.int32) - int(ts.zero_point)) * ts.scale

    print("====== Debug info ======")
    FORMAT_STR = "{:<16}{:<32}{:<24}{:<24}{:<32}{:<24}{:<24}{:<32}{:<24}{:<24}"
    print(
        logger.warning(
            FORMAT_STR.format(
                "op type",
                "input name",
                "input shape",
                "input range",
                "weight name",
                "weight shape",
                "weight range",
                "output name",
                "output shape",
                "output range",
            )
        )
    )

    def _print_stat(op, blobs, is_int8=False):
        stats = [op.type]
        for name in [op.input[0], op.input[1], op.output[0]]:
            stats.append(name)
            ts = blobs[name] if not is_int8 else _int8_to_fp(blobs[name])
            stats.append(str(ts.shape))
            stats.append("[{:8.3f}, {:8.3f}]".format(ts.min(), ts.max()))
        string = FORMAT_STR.format(*stats)
        if is_int8:
            print(logger.info(string))
        else:
            print(string)

    fp32_key_ops = [op for op in net.Proto().op if op.type in ["Conv", "ConvRelu"]]
    int8_key_ops = [
        op for op in int8_net.Proto().op if op.type in ["Int8Conv", "Int8ConvRelu"]
    ]

    # check matching
    assert len(fp32_key_ops) == len(int8_key_ops), "Mismatch between fp32/int8 net"

    for fp32_op, int8_op in zip(fp32_key_ops, int8_key_ops):
        _print_stat(fp32_op, FP32_BLOBS)
        _print_stat(int8_op, INT8_BLOBS, is_int8=True)


def save_q_net(
    int8_pred_net,
    int8_init_net,
    output_dir,
    blob_sizes=None,
    pred_model_name="model.pb",
    init_model_name="model_init.pb",
):
    if not PathManager.exists(output_dir):
        PathManager.mkdirs(output_dir)

    logger.info("save int8 model to {}".format(output_dir))

    with PathManager.open(output_dir + "/{}".format(pred_model_name), "wb") as f:
        f.write(int8_pred_net.SerializeToString())
    with PathManager.open(output_dir + "/{}txt".format(pred_model_name), "w") as f:
        f.write(str(int8_pred_net))
    with PathManager.open(output_dir + "/{}".format(init_model_name), "wb") as f:
        f.write(int8_init_net.SerializeToString())
    with PathManager.open(output_dir + "/{}txt".format(init_model_name), "w") as f:
        f.write(str(int8_init_net))

    # save model graph for easy visualization
    logger.info("Model def image saved to {}.".format(output_dir))
    save_graph(
        int8_pred_net,
        os.path.join(output_dir, "model_def.svg"),
        op_only=False,
        blob_sizes=blob_sizes
    )
    save_graph(
        int8_init_net,
        os.path.join(output_dir, "model_init_def.svg"),
        op_only=False,
        blob_sizes=blob_sizes
    )


def _append_order_switchin_and_de_quantize_op(fp32_predict_net, int8_predict_net):
    """
    Simply check the order and type of external outputs, add missing
        NHWC2NCHW and Int8Dequantize if necessary.
    """
    _, fp32_versions = core.get_ssa(fp32_predict_net)
    _, int8_versions = core.get_ssa(int8_predict_net)

    fp32_net_orders = analyze_order(fp32_predict_net)
    int8_net_orders = analyze_order(int8_predict_net)
    fp32_net_types = analyze_type(fp32_predict_net)
    int8_net_types = analyze_type(int8_predict_net)

    assert len(fp32_predict_net.external_output) == len(int8_predict_net.external_output) # noqa
    for ext_output_id in range(len(fp32_predict_net.external_output)):
        fp32_ext_output = fp32_predict_net.external_output[ext_output_id]
        int8_ext_output = int8_predict_net.external_output[ext_output_id]
        if int8_ext_output != fp32_ext_output:
            logger.warning("external output mismatch {} vs {}".format(
                fp32_ext_output, int8_ext_output))
            assert int8_ext_output.startswith(fp32_ext_output)

        fp32_versiond_output = (fp32_ext_output, fp32_versions[fp32_ext_output])
        fp32_net_order = fp32_net_orders[fp32_versiond_output]
        fp32_net_type = fp32_net_types[fp32_versiond_output]

        int8_versiond_output = (int8_ext_output, int8_versions[int8_ext_output])
        int8_net_order = int8_net_orders[int8_versiond_output]
        int8_net_type = int8_net_types[int8_versiond_output]

        # get the producer of corresponding external output, the int8_net_producer
        # since we're updating the int8_predict_net thoughout the process without
        # re-calculating ssa, thus we need to update int8_net_producer.
        int8_ssa, _ = core.get_ssa(int8_predict_net)
        int8_ssa_producer_map = get_producer_map(int8_ssa)
        int8_net_producer = int8_ssa_producer_map[int8_versiond_output]

        if (fp32_net_type != int8_net_type):
            logger.warning(
                "external output {} (in fp32 net) / {} (in int8 net) type mismatch,"
                " adding Int8Dequantize operator to fix".format(
                    fp32_ext_output, int8_ext_output
                )
            )
            assert fp32_net_type == "fp32" and int8_net_type == "int8"
            op_id, output_id = int8_net_producer
            producer_output = int8_predict_net.op[op_id].output[output_id]
            # original int8 net: ProducerOp -> producer_output
            # 1: add Int8Dequantize op to change producer_output to fp32_ext_output
            int8_predict_net.op.extend([
                core.CreateOperator(
                    "Int8Dequantize",
                    [producer_output],
                    [fp32_ext_output])
            ])
            int8_predict_net.external_output[ext_output_id] = fp32_ext_output
            int8_net_producer = (len(int8_predict_net.op) - 1, 0)
            # 2: rename producer's output name to [fp32_ext_output]_int8
            rename_op_output(
                int8_predict_net, op_id, output_id, fp32_ext_output + "_int8")
            # now the int8 net becomes:
            # ProducerOp  -> [fp32_ext_output]_NHWC -> NHWC2NCHW -> fp32_ext_output

        if (
            fp32_net_order is not None
            and int8_net_order is not None
            and fp32_net_order != int8_net_order
        ):
            logger.warning(
                "external output {} (in fp32 net) / {} (in int8 net) order mismatch,"
                " adding NHWC2NCHW operator to fix".format(
                    fp32_ext_output, int8_ext_output
                )
            )
            if not (fp32_net_order == "NCHW" and int8_net_order == "NHWC"):
                raise NotImplementedError("Only support this case now")
            op_id, output_id = int8_net_producer
            producer_output = int8_predict_net.op[op_id].output[output_id]
            # original int8 net: ProducerOp -> producer_output
            # 1: add NHWC2NCHW op to change producer_output to fp32_ext_output
            int8_predict_net.op.extend([
                core.CreateOperator("NHWC2NCHW", [producer_output], [fp32_ext_output])
            ])
            int8_predict_net.external_output[ext_output_id] = fp32_ext_output
            int8_net_producer = (len(int8_predict_net.op) - 1, 0)
            # 2: rename producer's output name to [fp32_ext_output]_NHWC
            rename_op_output(
                int8_predict_net, op_id, output_id, fp32_ext_output + "_NHWC")
            # now the int8 net becomes:
            # ProducerOp  -> [fp32_ext_output]_NHWC -> NHWC2NCHW -> fp32_ext_output

    # This should also fixed the name mismatch
    assert fp32_predict_net.external_output[ext_output_id] == \
        int8_predict_net.external_output[ext_output_id]


def main(
    cfg,
    output_dir,
    predict_net_path,
    init_net_path,
    runner=None,
    # binary specific optional arguments
    num_images_for_quant=1,
    quant_strat="MINMAX",
    skip_first_conv=False,
    ops_to_skip=None,
    show_int8_debug_info=False,
    fuse_relu=True,
    reset_engine=False,
    omit_im_info=False,
):
    setup_after_launch(cfg, output_dir, runner)

    logger.info("Loading Caffe2 model...")
    predict_net, init_net = _load_model(predict_net_path, init_net_path)
    pb_model = runner.build_caffe2_model(predict_net, init_net)
    cfg = pb_model.validate_cfg(cfg)

    data_loader = runner.build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    batched_inputs = []
    logger.info(
        "Loading first {} images used for calculate statistics for quantization ..."
        .format(num_images_for_quant)
    )
    for inputs in data_loader:
        batched_inputs.extend(inputs)
        if len(batched_inputs) >= num_images_for_quant:
            break
    batched_inputs = batched_inputs[:num_images_for_quant]

    # TODO: maybe pb_model should provide `get_caffe2_inputs` method, because
    # non-GeneralizedRCNN might have other way of doing this.
    from detectron2.export.caffe2_modeling import convert_batched_inputs_to_c2_format
    tensor_inputs = convert_batched_inputs_to_c2_format(
        batched_inputs,
        pb_model.size_divisibility,
        torch.device("cpu"),
    )
    # HACK: models other than detection may not need to feed im_info blob
    # this assumes converted model only needs tensor_input[0]
    if omit_im_info:
        tensor_inputs = tensor_inputs[:1]

    test_inputs = {
        name: tensor for name, tensor in zip(predict_net.external_input, tensor_inputs)
    }

    int8_predict_net, int8_init_net, fp32_blobs, int8_blobs = \
        convert_detection_model_to_int8(
            predict_net,
            init_net,
            test_inputs,
            quant_strat,
            skip_first_conv,
            ops_to_skip,
            fuse_relu,
            reset_engine,
        )

    logger.info("Adding NHWC2NCHW / Int8Dequantize if necessary ...")
    _append_order_switchin_and_de_quantize_op(predict_net, int8_predict_net)

    blob_sizes = {
        name: blob.shape if isinstance(blob, np.ndarray) else blob.data.shape
        for name, blob in int8_blobs.items()
        if isinstance(blob, (np.ndarray, workspace.Int8Tensor))
    }

    convert_output_folder = os.path.join(output_dir, "caffe2_int8_pb")
    save_q_net(int8_predict_net, int8_init_net, convert_output_folder, blob_sizes)

    if show_int8_debug_info:
        print("show int8 debug info....")
        raise NotImplementedError("Need a few fix to make int8_debug_info work")
        int8_debug_info(
            predict_net, init_net, fp32_blobs,
            int8_predict_net, int8_init_net, int8_blobs
        )

    # copy pre/post processing
    def _copy_if_exist(src_dir, dst_dir, name):
        src_file = os.path.join(src_dir, name)
        if PathManager.isfile(src_file):
            dst_file = os.path.join(dst_dir, name)
            PathManager.copy(src_file, dst_file, overwrite=True)

    model_dir = os.path.dirname(predict_net_path)
    _copy_if_exist(model_dir, convert_output_folder, "predictor_info.json")
    _copy_if_exist(model_dir, convert_output_folder, "preprocess.jit")
    _copy_if_exist(model_dir, convert_output_folder, "postprocess.jit")

    return {
        "predict_net_path": os.path.join(convert_output_folder, "model.pb"),
        "init_net_path": os.path.join(convert_output_folder, "model_init.pb"),
        "model_def_path": os.path.join(convert_output_folder, "model_def.svg"),
    }


@post_mortem_if_fail()
def run_locally(args):
    cfg, output_dir, runner = prepare_for_launch(args)
    return main(
        cfg,
        output_dir,
        os.path.join(args.input_dir, args.input_predict_net),
        os.path.join(args.input_dir, args.input_init_net),
        runner,
        # binary specific optional arguments
        args.num_images_for_quant,
        args.quant_strat,
        args.skip_first_conv,
        args.ops_to_skip,
        args.show_int8_debug_info,
        args.fuse_relu,
        args.reset_engine,
        args.omit_im_info,
    )


if __name__ == "__main__":
    parser = basic_argument_parser(distributed=False, requires_config_file=False)
    # === input/output path ====================================================
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Input model folder"
    )
    parser.add_argument(
        "--input_predict_net",
        type=str,
        default="model.pb",
        help="Input predict net name"
    )
    parser.add_argument(
        "--input_init_net",
        type=str,
        default="model_init.pb",
        help="Input init net name",
    )
    parser.add_argument(
        "--output_predict_net",
        type=str,
        default="model.pb",
        help="Output predict net name"
    )
    parser.add_argument(
        "--output_init_net",
        type=str,
        default="model_init.pb",
        help="Output init net name",
    )
    # === Int8 quantization configs ============================================
    parser.add_argument(
        "--num_images_for_quant",
        type=int,
        default=1,
        help="Number of batches used for collecting quantization statistics",
    )
    parser.add_argument(
        "--quant_strat",
        type=str,
        default="MINMAX",
        help="Quantization strategy, MINMAX, L2_APPROX, etc.",
    )
    parser.add_argument(
        "--skip_first_conv",
        default=0,
        type=int,
        help="If set, quantize first Conv/Relu/Pool as well",
    )
    parser.add_argument(
        "--ops_to_skip",
        nargs="+",
        type=str,
        default=[],
        help="List of ops to skip quantization",
    )
    parser.add_argument(
        "--show_int8_debug_info",
        type=int,
        default=0,
        help="show debug info such as blob range",
    )
    parser.add_argument(
        "--omit_im_info",
        action="store_true",
        help="omit im info blob when feeding inputs",
    )
    # === post quantization model editing ======================================
    parser.add_argument(
        "--fuse_relu", type=int, default=1, help="Fuse relu to conv if 1"
    )
    parser.add_argument(
        "--reset_engine",
        type=int,
        default=0,
        help="Reset all engine to empty if 1",
    )
    run_locally(parser.parse_args())
