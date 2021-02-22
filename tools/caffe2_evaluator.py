#!/usr/bin/env python3
"""
Binary to evaluate caffe2 models (represented as protobuf or logfile) using
detectron2 system (dataloading, evaluation, etc).
"""

import logging

from caffe2.proto import caffe2_pb2
from d2go.distributed import launch
from d2go.setup import (
    basic_argument_parser,
    caffe2_global_init,
    post_mortem_if_fail_for_main,
    prepare_for_launch,
    setup_after_launch,
)
from fvcore.common.file_io import PathManager


logger = logging.getLogger("d2go.tools.caffe2_evaluator")


def _load_model(predict_net_path, init_net_path, force_engine):
    predict_net = caffe2_pb2.NetDef()
    with PathManager.open(predict_net_path, "rb") as f:
        predict_net.ParseFromString(f.read())

    init_net = caffe2_pb2.NetDef()
    with PathManager.open(init_net_path, "rb") as f:
        init_net.ParseFromString(f.read())

    if force_engine is not None:
        for op in predict_net.op:
            op.engine = force_engine

    return predict_net, init_net


def main(
    cfg,
    output_dir,
    predict_net_path,
    init_net_path,
    runner=None,
    # binary specific optional arguments
    logging_print_net_summary=0,
    num_threads=None,
    force_engine=None,
):
    setup_after_launch(cfg, output_dir, runner)
    caffe2_global_init(logging_print_net_summary, num_threads)

    logger.info("Loading Caffe2 model...")
    predict_net, init_net = _load_model(predict_net_path, init_net_path, force_engine)
    pb_model = runner.build_caffe2_model(predict_net, init_net)
    cfg = pb_model.validate_cfg(cfg)

    # build normal test loader first to make sure metadata is backfilled
    for ds in cfg.DATASETS.TEST:
        runner.build_detection_test_loader(cfg, ds)
    accuracy = runner.do_test(cfg, pb_model)
    return {"accuracy": accuracy}


def run_with_cmdline_args(args):
    cfg, output_dir, runner = prepare_for_launch(args)
    launch(
        post_mortem_if_fail_for_main(main),
        args.num_processes,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        backend="GLOO",
        always_spawn=False,
        args=(
            cfg,
            output_dir,
            args.net,
            args.init_net,
            runner,
            # binary specific optional arguments
            args.logging_print_net_summary,
            args.num_threads,
            args.force_engine,
        ),
    )


if __name__ == "__main__":
    parser = basic_argument_parser(requires_config_file=False)
    parser.add_argument(
        "--net", dest="net", help="pb net path", default=None, type=str
    )
    parser.add_argument(
        "--init_net", dest="init_net", help="pb init net path", default=None, type=str
    )
    parser.add_argument(
        "--logfiledb", type=str, default=None, help="Path to the model file"
    )
    # === data loading =========================================================
    # parser.add_argument(
    #     "--mono_input", type=int, default=0, help="Convert input to mono if 1"
    # )
    # parser.add_argument(
    #     "--apply_meanstd",
    #     default=1,
    #     type=int,
    #     help="apply default mean/std to input image if 1",
    # )
    # === model modification ===================================================
    # parser.add_argument(
    #     "--score_thresh",
    #     type=float,
    #     default=0.0,
    #     help="Filter boxes below this score before evaluation",
    # )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Number of omp/mkl threads (per process) to use in Caffe2's GlobalInit",
    )
    parser.add_argument(
        "--force_engine",
        type=str,
        default=None,
        help="If set, engine of all ops will be set by this value",
    )
    # parser.add_argument(
    #     "--prof_dag",
    #     type=int,
    #     default=0,
    #     help="Benchmark each op",
    # )
    # === evaluation config ====================================================
    parser.add_argument(
        "--logging_print_net_summary",
        type=int,
        default=0,
        help="Control the --caffe2_logging_print_net_summary in GlobalInit",
    )
    run_with_cmdline_args(parser.parse_args())
