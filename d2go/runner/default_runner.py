#!/usr/bin/env python3

import contextlib
import logging
import math
import os
from collections import OrderedDict
from functools import lru_cache, partial
from typing import Type

import d2go.utils.abnormal_checker as abnormal_checker
import detectron2.utils.comm as comm
import mock
import torch
from d2go.config import CfgNode as CN, CONFIG_SCALING_METHOD_REGISTRY, temp_defrost, get_cfg_diff_table
from d2go.data.build import (
    build_weighted_detection_train_loader,
)
from d2go.data.dataset_mappers import build_dataset_mapper
from d2go.data.datasets import inject_coco_datasets, register_dynamic_datasets
from d2go.data.transforms.build import build_transform_gen
from d2go.data.utils import (
    maybe_subsample_n_images,
    update_cfg_if_using_adhoc_dataset,
)
from d2go.export.caffe2_model_helper import update_cfg_from_pb_model
from d2go.export.d2_meta_arch import patch_d2_meta_arch
from d2go.modeling import kmeans_anchors, model_ema
from d2go.modeling.model_freezing_utils import (
    set_requires_grad,
)
from d2go.modeling.quantization import (
    QATCheckpointer,
    setup_qat_model,
    silicon_qat_build_model_context,
)
from d2go.utils.flop_calculator import add_print_flops_callback
from d2go.utils.misc import get_tensorboard_log_dir
from d2go.utils.visualization import DataLoaderVisWrapper, VisualizationEvaluator
from d2go.utils.get_default_cfg import get_default_cfg
from d2go.optimizer import build_optimizer_mapper
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import (
    build_detection_test_loader as d2_build_detection_test_loader,
    build_detection_train_loader as d2_build_detection_train_loader,
    MetadataCatalog,
)
from detectron2.engine import SimpleTrainer, AMPTrainer, hooks
from detectron2.evaluation import (
    COCOEvaluator,
    RotatedCOCOEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.export.caffe2_inference import ProtobufDetectionModel
from detectron2.export.caffe2_modeling import META_ARCH_CAFFE2_EXPORT_TYPE_MAP
from d2go.utils.helper import TensorboardXWriter, D2Trainer
from detectron2.modeling import GeneralizedRCNNWithTTA, build_model
from detectron2.solver import (
    build_lr_scheduler as d2_build_lr_scheduler,
    build_optimizer as d2_build_optimizer,
)
from detectron2.utils.events import CommonMetricPrinter, JSONWriter
from mobile_cv.arch.quantization.observer import update_stat as observer_update_stat


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _mock_func(module, src_func, target_func):
    with mock.patch(
        "{}.{}".format(module.__name__, src_func.__name__), side_effect=target_func
    ) as mocked_func:
        yield
    if not mocked_func.call_count >= 1:
        logger.warning("Didn't patch the {} in module {}".format(src_func, module))


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
    max_iter = cfg.SOLVER.MAX_ITER
    steps = cfg.SOLVER.STEPS
    eval_period = cfg.TEST.EVAL_PERIOD
    ims_per_batch_train = cfg.SOLVER.IMS_PER_BATCH
    warmup_iters = cfg.SOLVER.WARMUP_ITERS

    # default configs in D2
    cfg.SOLVER.BASE_LR = base_lr * gpu_scale
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
    cfg.QUANTIZATION.QAT.DISABLE_OBSERVER_ITER = int(
        round(cfg.QUANTIZATION.QAT.DISABLE_OBSERVER_ITER / gpu_scale)
    )
    cfg.QUANTIZATION.QAT.FREEZE_BN_ITER = int(
        round(cfg.QUANTIZATION.QAT.FREEZE_BN_ITER / gpu_scale)
    )


class BaseRunner(object):
    def _initialize(self, cfg):
        """ Runner should be initialized in the sub-process in ddp setting """
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
        cfg = CN(cfg)  # upgrade from D2's CfgNode to D2Go's CfgNode
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

    def build_traceable_model(self, cfg, built_model=None):
        """
        Return a traceable model. The returned model has to be a
        `Caffe2MetaArch` which provides the following two member methods:
        - get_caffe2_inputs: it'll be called when exporting the model
            to convert D2's batched_input to a list of Tensors.
        - encode_additional_info: this allow editing exported predict_net/init_net.
        """
        return built_model

    def build_caffe2_model(self, predict_net, init_net):
        """
        Return a nn.Module which should behave the same as a normal D2 model.
        """
        raise NotImplementedError()

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

    def build_model(self, cfg, eval_only=False):
        # build_model might modify the cfg, thus clone
        cfg = cfg.clone()

        # silicon_qat_build_model_context is deprecated
        with silicon_qat_build_model_context(cfg):
            model = build_model(cfg)
            model_ema.may_build_model_ema(cfg, model)

        if cfg.MODEL.FROZEN_LAYER_REG_EXP:
            set_requires_grad(model, cfg.MODEL.FROZEN_LAYER_REG_EXP, False)

        if cfg.QUANTIZATION.QAT.ENABLED:
            # Disable fake_quant and observer so that the model will be trained normally
            # before QAT being turned on (controlled by QUANTIZATION.QAT.START_ITER).
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

    def build_checkpointer(self, cfg, model, save_dir, **kwargs):
        kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model))
        checkpointer = QATCheckpointer(model, save_dir=save_dir, **kwargs)
        return checkpointer

    def build_optimizer(self, cfg, model):
        return build_optimizer_mapper(cfg, model)

    def build_lr_scheduler(self, cfg, optimizer):
        return d2_build_lr_scheduler(cfg, optimizer)

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

        add_print_flops_callback(cfg, model, disable_after_callback=True)

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
                tbx_writer = _get_tbx_writer(get_tensorboard_log_dir(cfg.OUTPUT_DIR))
                logger.info("Adding visualization evaluator ...")
                mapper = self.get_mapper(cfg, is_train=False)
                evaluator._evaluators.append(
                    self.get_visualization_evaluator()(
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
                tbx_writer = _get_tbx_writer(get_tensorboard_log_dir(cfg.OUTPUT_DIR))
                tbx_writer._writer.add_scalar("eval_{}".format(k), v, train_iter)

        if comm.is_main_process():
            tbx_writer = _get_tbx_writer(get_tensorboard_log_dir(cfg.OUTPUT_DIR))
            tbx_writer._writer.flush()
        return results

    def do_test(self, cfg, model, train_iter=None):
        results = OrderedDict()
        with maybe_subsample_n_images(cfg) as new_cfg:
            # default model
            cur_results = self._do_test(
                new_cfg, model, train_iter=train_iter, model_tag="default"
            )
            results.update(cur_results)

            # model with ema weights
            if cfg.MODEL_EMA.ENABLED:
                logger.info("Run evaluation with EMA.")
                with model_ema.apply_model_ema_and_restore(model):
                    cur_results = self._do_test(
                        new_cfg, model, train_iter=train_iter, model_tag="ema"
                    )
                    results.update(cur_results)

        return results

    def do_train(self, cfg, model, resume):
        add_print_flops_callback(cfg, model, disable_after_callback=True)

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

            tbx_writer = _get_tbx_writer(get_tensorboard_log_dir(cfg.OUTPUT_DIR))
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
            tbx_writer = _get_tbx_writer(get_tensorboard_log_dir(cfg.OUTPUT_DIR))
            writers = [
                CommonMetricPrinter(max_iter),
                JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
                tbx_writer,
            ]
            trainer_hooks.append(hooks.PeriodicWriter(writers))
        trainer.register_hooks(trainer_hooks)
        trainer.train(start_iter, max_iter)

        if hasattr(self, 'original_cfg'):
            table = get_cfg_diff_table(cfg, self.original_cfg)
            logger.info("GeneralizeRCNN Runner ignoring training config change: \n" + table)
            trained_cfg = self.original_cfg.clone()
        else:
            trained_cfg = cfg.clone()
        with temp_defrost(trained_cfg):
            trained_cfg.MODEL.WEIGHTS = checkpointer.get_checkpoint_file()
        return {"model_final": trained_cfg}

    @classmethod
    def build_detection_test_loader(cls, cfg, dataset_name, mapper=None):
        logger.info(
            "Building detection test loader for dataset: {} ...".format(dataset_name)
        )
        mapper = mapper or cls.get_mapper(cfg, is_train=False)
        logger.info("Using dataset mapper:\n{}".format(mapper))
        return d2_build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_detection_train_loader(cls, cfg, *args, mapper=None, **kwargs):
        logger.info("Building detection train loader ...")
        mapper = mapper or cls.get_mapper(cfg, is_train=True)
        logger.info("Using dataset mapper:\n{}".format(mapper))

        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        if sampler_name == "WeightedTrainingSampler":
            data_loader = build_weighted_detection_train_loader(cfg, mapper=mapper)
        else:
            data_loader = d2_build_detection_train_loader(
                cfg, *args, mapper=mapper, **kwargs
            )

        if comm.is_main_process():
            tbx_writer = _get_tbx_writer(get_tensorboard_log_dir(cfg.OUTPUT_DIR))
            data_loader = cls.get_data_loader_vis_wrapper()(
                cfg, tbx_writer, data_loader
            )
        return data_loader

    @staticmethod
    def get_data_loader_vis_wrapper() -> Type[DataLoaderVisWrapper]:
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
            )
        elif evaluator_type in ["rotated_coco"]:
            dataset_evaluators = DatasetEvaluators(
                [RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)]
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
    def get_visualization_evaluator() -> Type[VisualizationEvaluator]:
        return VisualizationEvaluator

    @staticmethod
    def final_model_name():
        return "model_final"

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

    def _create_qat_hook(self, cfg):
        """
        Create a hook to start QAT (during training) and/or change the phase of QAT.
        """
        applied = {
            "enable_fake_quant": False,
            "enable_observer": False,
            "disable_observer": False,
            "freeze_bn_stats": False,
        }

        assert (
            cfg.QUANTIZATION.QAT.ENABLE_OBSERVER_ITER
            <= cfg.QUANTIZATION.QAT.DISABLE_OBSERVER_ITER
        ), "Can't diable observer before enabling it"

        def qat_before_step_callback(trainer):
            if (
                not applied["enable_fake_quant"]
                and trainer.iter >= cfg.QUANTIZATION.QAT.START_ITER
            ):
                logger.info(
                    "[QAT] enable fake quant to start QAT, iter = {}".format(
                        trainer.iter
                    )
                )
                trainer.model.apply(torch.quantization.enable_fake_quant)
                applied["enable_fake_quant"] = True

                if cfg.QUANTIZATION.QAT.BATCH_SIZE_FACTOR != 1.0:
                    loader_cfg = cfg.clone()
                    loader_cfg.defrost()
                    num_gpus = comm.get_world_size()
                    old_bs = cfg.SOLVER.IMS_PER_BATCH // num_gpus
                    new_bs = math.ceil(old_bs * cfg.QUANTIZATION.QAT.BATCH_SIZE_FACTOR)
                    loader_cfg.SOLVER.IMS_PER_BATCH = new_bs * num_gpus
                    loader_cfg.freeze()

                    logger.info(
                        "[QAT] Rebuild data loader with batch size per GPU: {} -> {}".format(
                            old_bs, new_bs
                        )
                    )
                    # This method assumes the data loader can be replaced from trainer
                    assert trainer.__class__ == SimpleTrainer
                    del trainer._data_loader_iter
                    del trainer.data_loader
                    data_loader = self.build_detection_train_loader(loader_cfg)
                    trainer.data_loader = data_loader
                    trainer._data_loader_iter = iter(data_loader)

            if (
                not applied["enable_observer"]
                and trainer.iter >= cfg.QUANTIZATION.QAT.ENABLE_OBSERVER_ITER
                and trainer.iter < cfg.QUANTIZATION.QAT.DISABLE_OBSERVER_ITER
            ):
                logger.info("[QAT] enable observer, iter = {}".format(trainer.iter))
                trainer.model.apply(torch.quantization.enable_observer)
                applied["enable_observer"] = True

            if (
                not applied["disable_observer"]
                and trainer.iter >= cfg.QUANTIZATION.QAT.DISABLE_OBSERVER_ITER
            ):
                logger.info(
                    "[QAT] disabling observer for sub seq iters, iter = {}".format(
                        trainer.iter
                    )
                )
                trainer.model.apply(torch.quantization.disable_observer)
                applied["disable_observer"] = True

            if (
                not applied["freeze_bn_stats"]
                and trainer.iter >= cfg.QUANTIZATION.QAT.FREEZE_BN_ITER
            ):
                logger.info(
                    "[QAT] freezing BN for subseq iters, iter = {}".format(trainer.iter)
                )
                trainer.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                applied["freeze_bn_stats"] = True

            if (
                applied["enable_fake_quant"]
                and cfg.QUANTIZATION.QAT.UPDATE_OBSERVER_STATS_PERIODICALLY
                and trainer.iter % cfg.QUANTIZATION.QAT.UPDATE_OBSERVER_STATS_PERIOD
                == 0
            ):
                logger.info(f"[QAT] updating observers, iter = {trainer.iter}")
                trainer.model.apply(observer_update_stat)

        return hooks.CallbackHook(before_step=qat_before_step_callback)


class GeneralizedRCNNRunner(Detectron2GoRunner):
    @staticmethod
    def get_default_cfg():
        _C = super(GeneralizedRCNNRunner, GeneralizedRCNNRunner).get_default_cfg()
        _C.EXPORT_CAFFE2 = CN()
        _C.EXPORT_CAFFE2.USE_HEATMAP_MAX_KEYPOINT = False
        return _C

    def build_traceable_model(self, cfg, built_model=None):
        if built_model is not None:
            logger.warning("The given built_model will be modified")
        else:
            built_model = self.build_model(cfg, eval_only=True)
            logger.info("Model:\n{}".format(built_model))

        Caffe2ModelType = META_ARCH_CAFFE2_EXPORT_TYPE_MAP[cfg.MODEL.META_ARCHITECTURE]
        return Caffe2ModelType(cfg, torch_model=built_model)

    def build_caffe2_model(self, predict_net, init_net):
        pb_model = ProtobufDetectionModel(predict_net, init_net)
        pb_model.validate_cfg = partial(update_cfg_from_pb_model, model=pb_model)
        return pb_model
