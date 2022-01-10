#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import os
from collections import OrderedDict
from functools import lru_cache
from typing import Type, Optional, List, Union

import d2go.utils.abnormal_checker as abnormal_checker
import detectron2.utils.comm as comm
import torch
from d2go.config import (
    CfgNode,
    CONFIG_SCALING_METHOD_REGISTRY,
    temp_defrost,
)
from d2go.config.utils import get_cfg_diff_table
from d2go.data.build import build_d2go_train_loader
from d2go.data.dataset_mappers import build_dataset_mapper
from d2go.data.datasets import inject_coco_datasets, register_dynamic_datasets
from d2go.data.transforms.build import build_transform_gen
from d2go.data.utils import (
    maybe_subsample_n_images,
    update_cfg_if_using_adhoc_dataset,
)
from d2go.export.d2_meta_arch import patch_d2_meta_arch
from d2go.modeling import kmeans_anchors, model_ema
from d2go.modeling.model_freezing_utils import (
    freeze_matched_bn,
    set_requires_grad,
)
from d2go.modeling.quantization import (
    QATCheckpointer,
    setup_qat_model,
    QATHook,
)
from d2go.optimizer import build_optimizer_mapper
from d2go.utils.flop_calculator import add_flop_printing_hook
from d2go.utils.get_default_cfg import get_default_cfg
from d2go.utils.helper import TensorboardXWriter, D2Trainer
from d2go.utils.misc import get_tensorboard_log_dir
from d2go.utils.visualization import DataLoaderVisWrapper, VisualizationEvaluator
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import (
    build_detection_test_loader as d2_build_detection_test_loader,
    build_detection_train_loader as d2_build_detection_train_loader,
    MetadataCatalog,
)
from detectron2.engine import HookBase
from detectron2.engine import SimpleTrainer, AMPTrainer, hooks
from detectron2.evaluation import (
    COCOEvaluator,
    RotatedCOCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA, build_model
from detectron2.solver import (
    build_lr_scheduler as d2_build_lr_scheduler,
)
from detectron2.utils.events import CommonMetricPrinter, JSONWriter
from detectron2.utils.registry import Registry
from mobile_cv.predictor.api import PredictorWrapper


logger = logging.getLogger(__name__)


ALL_TB_WRITERS = []


@lru_cache()
def _get_tbx_writer(log_dir):
    ret = TensorboardXWriter(log_dir)
    ALL_TB_WRITERS.append(ret)
    return ret


def _close_all_tbx_writers():
    for x in ALL_TB_WRITERS:
        x.close()
    ALL_TB_WRITERS.clear()


@CONFIG_SCALING_METHOD_REGISTRY.register()
def default_scale_d2_configs(cfg, new_world_size):
    gpu_scale = new_world_size / cfg.SOLVER.REFERENCE_WORLD_SIZE

    base_lr = cfg.SOLVER.BASE_LR
    base_lr_end = cfg.SOLVER.BASE_LR_END
    max_iter = cfg.SOLVER.MAX_ITER
    steps = cfg.SOLVER.STEPS
    eval_period = cfg.TEST.EVAL_PERIOD
    ims_per_batch_train = cfg.SOLVER.IMS_PER_BATCH
    warmup_iters = cfg.SOLVER.WARMUP_ITERS

    # lr scale
    lr_scales = {
        "sgd": gpu_scale,
        "adamw": 1,
        "sgd_mt": gpu_scale,
        "adamw_mt": 1,
    }
    optim_name = cfg.SOLVER.OPTIMIZER.lower()
    lr_scale = lr_scales[optim_name] if optim_name in lr_scales else gpu_scale

    # default configs in D2
    cfg.SOLVER.BASE_LR = base_lr * lr_scale
    cfg.SOLVER.BASE_LR_END = base_lr_end * lr_scale
    cfg.SOLVER.MAX_ITER = int(round(max_iter / gpu_scale))
    cfg.SOLVER.STEPS = tuple(int(round(s / gpu_scale)) for s in steps)
    cfg.TEST.EVAL_PERIOD = int(round(eval_period / gpu_scale))
    cfg.SOLVER.IMS_PER_BATCH = int(round(ims_per_batch_train * gpu_scale))
    cfg.SOLVER.WARMUP_ITERS = int(round(warmup_iters / gpu_scale))


@CONFIG_SCALING_METHOD_REGISTRY.register()
def default_scale_quantization_configs(cfg, new_world_size):
    gpu_scale = new_world_size / cfg.SOLVER.REFERENCE_WORLD_SIZE

    # Scale QUANTIZATION related configs
    cfg.QUANTIZATION.QAT.START_ITER = int(
        round(cfg.QUANTIZATION.QAT.START_ITER / gpu_scale)
    )
    cfg.QUANTIZATION.QAT.ENABLE_OBSERVER_ITER = int(
        round(cfg.QUANTIZATION.QAT.ENABLE_OBSERVER_ITER / gpu_scale)
    )
    cfg.QUANTIZATION.QAT.ENABLE_LEARNABLE_OBSERVER_ITER = int(
        round(cfg.QUANTIZATION.QAT.ENABLE_LEARNABLE_OBSERVER_ITER / gpu_scale)
    )
    cfg.QUANTIZATION.QAT.DISABLE_OBSERVER_ITER = int(
        round(cfg.QUANTIZATION.QAT.DISABLE_OBSERVER_ITER / gpu_scale)
    )
    cfg.QUANTIZATION.QAT.FREEZE_BN_ITER = int(
        round(cfg.QUANTIZATION.QAT.FREEZE_BN_ITER / gpu_scale)
    )


class BaseRunner(object):
    def __init__(self):
        identifier = f"D2Go.Runner.{self.__class__.__name__}"
        torch._C._log_api_usage_once(identifier)

    def _initialize(self, cfg):
        """Runner should be initialized in the sub-process in ddp setting"""
        if getattr(self, "_has_initialized", False):
            logger.warning("Runner has already been initialized, skip initialization.")
            return
        self._has_initialized = True
        self.register(cfg)

    def register(self, cfg):
        """
        Override `register` in order to run customized code before other things like:
            - registering datasets.
            - registering model using Registry.
        """
        pass

    @staticmethod
    def get_default_cfg():
        """
        Override `get_default_cfg` for adding non common config.
        """
        from detectron2.config import get_cfg as get_d2_cfg

        cfg = get_d2_cfg()
        cfg = CfgNode.cast_from_other_class(
            cfg
        )  # upgrade from D2's CfgNode to D2Go's CfgNode

        cfg.SOLVER.AUTO_SCALING_METHODS = ["default_scale_d2_configs"]
        return cfg

    def build_model(self, cfg, eval_only=False):
        # cfg may need to be reused to build trace model again, thus clone
        model = build_model(cfg.clone())

        if eval_only:
            checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            model.eval()

        return model

    def do_test(self, *args, **kwargs):
        raise NotImplementedError()

    def do_train(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def build_detection_test_loader(cls, *args, **kwargs):
        return d2_build_detection_test_loader(*args, **kwargs)

    @classmethod
    def build_detection_train_loader(cls, *args, **kwargs):
        return d2_build_detection_train_loader(*args, **kwargs)


# List of functions to add hooks for trainer, all functions in the registry will
# be called to add hooks
#   func(hooks: List[HookBase]) -> None
TRAINER_HOOKS_REGISTRY = Registry("TRAINER_HOOKS_REGISTRY")


def update_hooks_from_registry(hooks: List[HookBase]):
    for name, hook_func in TRAINER_HOOKS_REGISTRY:
        logger.info(f"Update trainer hooks from {name}...")
        hook_func(hooks)


class Detectron2GoRunner(BaseRunner):
    def register(self, cfg):
        super().register(cfg)
        self.original_cfg = cfg.clone()
        inject_coco_datasets(cfg)
        register_dynamic_datasets(cfg)
        update_cfg_if_using_adhoc_dataset(cfg)
        patch_d2_meta_arch()

    @staticmethod
    def get_default_cfg():
        _C = super(Detectron2GoRunner, Detectron2GoRunner).get_default_cfg()
        return get_default_cfg(_C)

    # temporary API
    def _build_model(self, cfg, eval_only=False):
        # build_model might modify the cfg, thus clone
        cfg = cfg.clone()

        model = build_model(cfg)
        model_ema.may_build_model_ema(cfg, model)

        if cfg.MODEL.FROZEN_LAYER_REG_EXP:
            set_requires_grad(model, cfg.MODEL.FROZEN_LAYER_REG_EXP, False)
            model = freeze_matched_bn(model, cfg.MODEL.FROZEN_LAYER_REG_EXP)

        if cfg.QUANTIZATION.QAT.ENABLED:
            # Disable fake_quant and observer so that the model will be trained normally
            # before QAT being turned on (controlled by QUANTIZATION.QAT.START_ITER).
            if hasattr(model, "get_rand_input"):
                model = setup_qat_model(
                    cfg, model, enable_fake_quant=eval_only, enable_observer=True
                )
                imsize = cfg.INPUT.MAX_SIZE_TRAIN
                rand_input = model.get_rand_input(imsize)
                model(rand_input, {})
            else:
                model = setup_qat_model(
                    cfg, model, enable_fake_quant=eval_only, enable_observer=False
                )

        if eval_only:
            checkpointer = self.build_checkpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            model.eval()

            if cfg.MODEL_EMA.ENABLED and cfg.MODEL_EMA.USE_EMA_WEIGHTS_FOR_EVAL_ONLY:
                model_ema.apply_model_ema(model)

        return model

    def build_model(self, cfg, eval_only=False):
        model = self._build_model(cfg, eval_only)

        # Note: the _visualize_model API is experimental
        if comm.is_main_process():
            if hasattr(model, "_visualize_model"):
                logger.info("Adding model visualization ...")
                tbx_writer = self.get_tbx_writer(cfg)
                model._visualize_model(tbx_writer)

        return model

    def build_checkpointer(self, cfg, model, save_dir, **kwargs):
        kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model))
        checkpointer = QATCheckpointer(model, save_dir=save_dir, **kwargs)
        return checkpointer

    def build_optimizer(self, cfg, model):
        return build_optimizer_mapper(cfg, model)

    def build_lr_scheduler(self, cfg, optimizer):
        return d2_build_lr_scheduler(cfg, optimizer)

    @classmethod
    def get_tbx_writer(cls, cfg):
        return _get_tbx_writer(get_tensorboard_log_dir(cfg.OUTPUT_DIR))

    def _do_test(self, cfg, model, train_iter=None, model_tag="default"):
        """train_iter: Current iteration of the model, None means final iteration"""
        assert len(cfg.DATASETS.TEST)
        assert cfg.OUTPUT_DIR

        is_final = (train_iter is None) or (train_iter == cfg.SOLVER.MAX_ITER - 1)

        logger.info(
            f"Running evaluation for model tag {model_tag} at iter {train_iter}..."
        )

        def _get_inference_dir_name(base_dir, inference_type, dataset_name):
            return os.path.join(
                base_dir,
                inference_type,
                model_tag,
                str(train_iter) if train_iter is not None else "final",
                dataset_name,
            )

        add_flop_printing_hook(model, cfg.OUTPUT_DIR)

        results = OrderedDict()
        results[model_tag] = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            # Evaluator will create output folder, no need to create here
            output_folder = _get_inference_dir_name(
                cfg.OUTPUT_DIR, "inference", dataset_name
            )

            # NOTE: creating evaluator after dataset is loaded as there might be dependency.  # noqa
            data_loader = self.build_detection_test_loader(cfg, dataset_name)
            evaluator = self.get_evaluator(
                cfg, dataset_name, output_folder=output_folder
            )

            if not isinstance(evaluator, DatasetEvaluators):
                evaluator = DatasetEvaluators([evaluator])
            if comm.is_main_process():
                tbx_writer = self.get_tbx_writer(cfg)
                logger.info("Adding visualization evaluator ...")
                mapper = self.get_mapper(cfg, is_train=False)
                vis_eval_type = self.get_visualization_evaluator()
                if vis_eval_type is not None:
                    evaluator._evaluators.append(
                        vis_eval_type(
                            cfg,
                            tbx_writer,
                            mapper,
                            dataset_name,
                            train_iter=train_iter,
                            tag_postfix=model_tag,
                        )
                    )

            results_per_dataset = inference_on_dataset(model, data_loader, evaluator)

            if comm.is_main_process():
                results[model_tag][dataset_name] = results_per_dataset
                if is_final:
                    print_csv_format(results_per_dataset)

            if is_final and cfg.TEST.AUG.ENABLED:
                # In the end of training, run an evaluation with TTA
                # Only support some R-CNN models.
                output_folder = _get_inference_dir_name(
                    cfg.OUTPUT_DIR, "inference_TTA", dataset_name
                )

                logger.info("Running inference with test-time augmentation ...")
                data_loader = self.build_detection_test_loader(
                    cfg, dataset_name, mapper=lambda x: x
                )
                evaluator = self.get_evaluator(
                    cfg, dataset_name, output_folder=output_folder
                )
                inference_on_dataset(
                    GeneralizedRCNNWithTTA(cfg, model), data_loader, evaluator
                )

        if is_final and cfg.TEST.EXPECTED_RESULTS and comm.is_main_process():
            assert len(results) == 1, "Results verification only supports one dataset!"
            verify_results(cfg, results[model_tag][cfg.DATASETS.TEST[0]])

        # write results to tensorboard
        if comm.is_main_process() and results:
            from detectron2.evaluation.testing import flatten_results_dict

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                tbx_writer = self.get_tbx_writer(cfg)
                tbx_writer._writer.add_scalar("eval_{}".format(k), v, train_iter)

        if comm.is_main_process():
            tbx_writer = self.get_tbx_writer(cfg)
            tbx_writer._writer.flush()
        return results

    def do_test(self, cfg, model, train_iter=None):
        """do_test does not load the weights of the model.
        If you want to use it outside the regular training routine,
        you will have to load the weights through a checkpointer.
        """
        results = OrderedDict()
        with maybe_subsample_n_images(cfg) as new_cfg:
            # default model
            cur_results = self._do_test(
                new_cfg, model, train_iter=train_iter, model_tag="default"
            )
            results.update(cur_results)

            # model with ema weights
            if cfg.MODEL_EMA.ENABLED and not isinstance(model, PredictorWrapper):
                logger.info("Run evaluation with EMA.")
                with model_ema.apply_model_ema_and_restore(model):
                    cur_results = self._do_test(
                        new_cfg, model, train_iter=train_iter, model_tag="ema"
                    )
                    results.update(cur_results)

        return results

    def do_train(self, cfg, model, resume):
        # Note that flops at the beginning of training is often inaccurate,
        # if a model has input-dependent logic
        add_flop_printing_hook(model, cfg.OUTPUT_DIR)

        optimizer = self.build_optimizer(cfg, model)
        scheduler = self.build_lr_scheduler(cfg, optimizer)

        checkpointer = self.build_checkpointer(
            cfg,
            model,
            save_dir=cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
        start_iter = (
            checkpoint.get("iteration", -1)
            if resume and checkpointer.has_checkpoint()
            else -1
        )
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        start_iter += 1
        max_iter = cfg.SOLVER.MAX_ITER
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )

        data_loader = self.build_detection_train_loader(cfg)

        def _get_model_with_abnormal_checker(model):
            if not cfg.ABNORMAL_CHECKER.ENABLED:
                return model

            tbx_writer = self.get_tbx_writer(cfg)
            writers = abnormal_checker.get_writers(cfg, tbx_writer)
            checker = abnormal_checker.AbnormalLossChecker(start_iter, writers)
            ret = abnormal_checker.AbnormalLossCheckerWrapper(model, checker)
            return ret

        trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            _get_model_with_abnormal_checker(model), data_loader, optimizer
        )
        trainer_hooks = [
            hooks.IterationTimer(),
            model_ema.EMAHook(cfg, model) if cfg.MODEL_EMA.ENABLED else None,
            self._create_data_loader_hook(cfg),
            self._create_after_step_hook(
                cfg, model, optimizer, scheduler, periodic_checkpointer
            ),
            hooks.EvalHook(
                cfg.TEST.EVAL_PERIOD,
                lambda: self.do_test(cfg, model, train_iter=trainer.iter),
            ),
            kmeans_anchors.compute_kmeans_anchors_hook(self, cfg),
            self._create_qat_hook(cfg) if cfg.QUANTIZATION.QAT.ENABLED else None,
        ]

        if comm.is_main_process():
            tbx_writer = self.get_tbx_writer(cfg)
            writers = [
                CommonMetricPrinter(max_iter),
                JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
                tbx_writer,
            ]
            trainer_hooks.append(hooks.PeriodicWriter(writers))
        update_hooks_from_registry(trainer_hooks)
        trainer.register_hooks(trainer_hooks)
        trainer.train(start_iter, max_iter)

        if hasattr(self, "original_cfg"):
            table = get_cfg_diff_table(cfg, self.original_cfg)
            logger.info(
                "GeneralizeRCNN Runner ignoring training config change: \n" + table
            )
            trained_cfg = self.original_cfg.clone()
        else:
            trained_cfg = cfg.clone()
        with temp_defrost(trained_cfg):
            trained_cfg.MODEL.WEIGHTS = checkpointer.get_checkpoint_file()
        return {"model_final": trained_cfg}

    @classmethod
    def build_detection_test_loader(
        cls, cfg, dataset_name: Union[str, List[str]], mapper=None
    ):
        logger.info(
            "Building detection test loader for dataset: {} ...".format(dataset_name)
        )
        mapper = mapper or cls.get_mapper(cfg, is_train=False)
        logger.info("Using dataset mapper:\n{}".format(mapper))
        return d2_build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_detection_train_loader(cls, cfg, *args, mapper=None, **kwargs):
        mapper = mapper or cls.get_mapper(cfg, is_train=True)
        data_loader = build_d2go_train_loader(cfg, mapper)
        return cls._attach_visualizer_to_data_loader(cfg, data_loader)

    @staticmethod
    def get_data_loader_vis_wrapper() -> Optional[Type[DataLoaderVisWrapper]]:
        return DataLoaderVisWrapper

    @staticmethod
    def get_evaluator(cfg, dataset_name, output_folder):
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            # D2 is in the process of reducing the use of cfg.
            dataset_evaluators = COCOEvaluator(
                dataset_name,
                output_dir=output_folder,
                kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS,
                max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
            )
        elif evaluator_type in ["rotated_coco"]:
            dataset_evaluators = DatasetEvaluators(
                [RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)]
            )
        elif evaluator_type in ["lvis"]:
            dataset_evaluators = LVISEvaluator(
                dataset_name,
                output_dir=output_folder,
                max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
            )
        else:
            dataset_evaluators = D2Trainer.build_evaluator(
                cfg, dataset_name, output_folder
            )
        if not isinstance(dataset_evaluators, DatasetEvaluators):
            dataset_evaluators = DatasetEvaluators([dataset_evaluators])
        return dataset_evaluators

    @staticmethod
    def get_mapper(cfg, is_train):
        tfm_gens = build_transform_gen(cfg, is_train)
        mapper = build_dataset_mapper(cfg, is_train, tfm_gens=tfm_gens)
        return mapper

    @staticmethod
    def get_visualization_evaluator() -> Optional[Type[VisualizationEvaluator]]:
        return VisualizationEvaluator

    @staticmethod
    def final_model_name():
        return "model_final"

    @classmethod
    def _attach_visualizer_to_data_loader(cls, cfg, data_loader):
        if comm.is_main_process():
            data_loader_type = cls.get_data_loader_vis_wrapper()
            if data_loader_type is not None:
                tbx_writer = cls.get_tbx_writer(cfg)
                data_loader = data_loader_type(cfg, tbx_writer, data_loader)
        return data_loader

    def _create_after_step_hook(
        self, cfg, model, optimizer, scheduler, periodic_checkpointer
    ):
        """
        Create a hook that performs some pre-defined tasks used in this script
        (evaluation, LR scheduling, checkpointing).
        """

        def after_step_callback(trainer):
            trainer.storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
            )
            scheduler.step()
            # Note: when precise BN is enabled, some checkpoints will have more precise
            # statistics than others, if they are saved immediately after eval.
            if comm.is_main_process():
                periodic_checkpointer.step(trainer.iter)

        return hooks.CallbackHook(after_step=after_step_callback)

    def _create_data_loader_hook(self, cfg):
        """
        Create a hook for manipulating data loader
        """
        return None

    def _create_qat_hook(self, cfg) -> Optional[QATHook]:
        """
        Create a hook to start QAT (during training) and/or change the phase of QAT.
        """
        if not cfg.QUANTIZATION.QAT.ENABLED:
            return None

        return QATHook(cfg, self.build_detection_train_loader)


def _add_rcnn_default_config(_C):
    _C.EXPORT_CAFFE2 = CfgNode()
    _C.EXPORT_CAFFE2.USE_HEATMAP_MAX_KEYPOINT = False

    _C.RCNN_PREPARE_FOR_EXPORT = "default_rcnn_prepare_for_export"
    _C.RCNN_PREPARE_FOR_QUANT = "default_rcnn_prepare_for_quant"
    _C.RCNN_PREPARE_FOR_QUANT_CONVERT = "default_rcnn_prepare_for_quant_convert"


class GeneralizedRCNNRunner(Detectron2GoRunner):
    @staticmethod
    def get_default_cfg():
        _C = super(GeneralizedRCNNRunner, GeneralizedRCNNRunner).get_default_cfg()
        _add_rcnn_default_config(_C)
        return _C
