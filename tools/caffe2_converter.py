#!/usr/bin/env python3
"""
Binary to convert pytorch detectron2 model to caffe2 model.
"""

import logging
import os

import torch
from d2go.export.caffe2 import export_caffe2
from d2go.export.torchscript import trace_and_save_torchscript
from d2go.setup import (
    basic_argument_parser,
    prepare_for_launch,
    setup_after_launch,
)
from mobile_cv.common.misc.py import post_mortem_if_fail


logger = logging.getLogger("d2go.tools.caffe2_converter")


def _print_csv_format_multiple_results(names, results_list):
    """
    like print_csv_format, but take list of names and results.
    """
    import collections

    # unordered results cannot be properly printed
    assert all(isinstance(x, collections.OrderedDict) for x in results_list)
    results_tasks = results_list[0].keys()
    assert all(x.keys() == results_tasks for x in results_list)

    max_length = max(len(x) for x in names)
    aligned_names = ["{0:>{1}}".format(x, max_length) for x in names]

    for task in [
        "bbox",
        "segm",
        "keypoints",
        "box_proposals",
        "sem_seg",
        "panoptic_seg",
    ]:
        if task not in results_tasks:
            continue
        res_list = [r[task] for r in results_list]
        metrics = res_list[0].keys()
        assert all(r.keys() == metrics for r in res_list)
        logger.info("copypaste: Task: {}".format(task))
        logger.info("copypaste: " + ",".join(["name"] + list(metrics)))
        for name, res in zip(aligned_names, res_list):
            to_show = ["{0:.4f}".format(v) for v in res.values()]
            logger.info("copypaste: " + ",".join([name] + to_show))


def _save_torch_script(caffe2_compatible_model, inputs, output_path):
    logger.info("Exporting torch script model to {} ...".format(output_path))
    with torch.no_grad():
        script_model = torch.jit.trace(caffe2_compatible_model, (inputs,))
    script_model.save(output_path)


def main(
    cfg,
    output_dir,
    runner=None,
    # binary specific optional arguments
    save_logdb=False,
    compare_accuracy=False,
):
    setup_after_launch(cfg, output_dir, runner)

    # this model takes tensors as input/output and thus tracable
    traceable_model = runner.build_traceable_model(cfg)
    # prepare data
    data_loader = runner.build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    first_batch = next(iter(data_loader))
    tensor_inputs = traceable_model.get_caffe2_inputs(first_batch)

    # export caffe2
    caffe2_model, caffe2_export_paths = export_caffe2(
        traceable_model,
        tensor_inputs,
        output_dir=os.path.join(output_dir, "caffe2_pb"),
        save_logdb=save_logdb,
    )

    # export torch script
    torch_script_path = os.path.join(output_dir, "model.pth")
    trace_and_save_torchscript(traceable_model, (tensor_inputs,), torch_script_path)

    if compare_accuracy:
        torch_model = runner.build_model(cfg, eval_only=True)
        pb_results = runner.do_test(cfg, caffe2_model)
        torch_results = runner.do_test(cfg, torch_model)

        logger.info("Summarizing accuracy comparison")
        for pb_results_per_ds, torch_results_per_ds in zip(
            pb_results.values(), torch_results.values()
        ):
            _print_csv_format_multiple_results(
                names=["caffe2", "pytorch"],
                results_list=[pb_results_per_ds, torch_results_per_ds],
            )

    # TODO what if it fails in the middle, shall/can we return partial results?
    ret = {"torch_script_path": torch_script_path, **caffe2_export_paths}
    # TODO return more things if needed, eg. accuracy, etc.
    return ret


@post_mortem_if_fail()
def run_with_cmdline_args(args):
    cfg, output_dir, runner = prepare_for_launch(args)
    return main(
        cfg,
        output_dir,
        runner,
        # binary specific optional arguments
        args.save_logdb,
        args.compare_accuracy,
    )


if __name__ == "__main__":
    parser = basic_argument_parser(distributed=False)
    parser.add_argument(
        "--save_logdb", default=0, type=int, help="Save output model to logdb format"
    )
    parser.add_argument(
        "--compare-accuracy",
        action="store_true",
        help="If true, both original and converted model will be evaluted on "
        "cfg.DATASETS.TEST",
    )
    run_with_cmdline_args(parser.parse_args())
