#!/usr/bin/env python3

import unittest
import itertools

import torch
from common.utils_pytorch.model_utils import has_module
from d2go.config import CfgNode as CN
from d2go.export.api import convert_and_export_predictor
from d2go.modeling.backbone.fbnet_cfg import add_fbnet_v2_default_configs
from d2go.modeling.meta_arch.regressor import (
    REGRESSOR_HEADS_REGISTRY,
    REGRESSOR_LOSS_REGISTRY,
    DeployableModel,
    Regressor,
    add_regressor_default_configs,
)
from d2go.modeling.quantization import (
    add_quantization_default_configs,
    post_training_quantize,
    setup_qat_model,
)
from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.predictor.api import PredictorWrapper, create_predictor
from oculus.face.social_eye.lib.loss import (
    SmoothL1OutlierRejectionLoss,
    add_smooth_l1_outlier_rejection_loss_default_configs,
)
from oculus.face.social_eye.lib.model.heads import (
    GazeHead,
    add_gaze_head_default_configs,
)
from oculus.face.social_eye.lib.model.resnet_backbone import (
    SplitAndConcat,
    init_weights,
)
from torch.quantization import DeQuantStub, QuantStub


REGRESSOR_HEADS_REGISTRY.register(GazeHead)
REGRESSOR_LOSS_REGISTRY.register(SmoothL1OutlierRejectionLoss)


class DummyMultipleHead(torch.nn.Module):
    """Test head where the head itself has a dict output with multiple outputs"""

    def __init__(self, cfg, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 4
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # split into 2 chunks along batch dimension and concat along channel
        # dimension: 2N x C x H x W -> N x 2C x H x W
        self.splitconcat = SplitAndConcat(split_dim=0, concat_dim=1)
        self.fcx = torch.nn.Linear(2 * in_channels, self.out_channels)
        self.fcy = torch.nn.Linear(2 * in_channels, self.out_channels)
        init_weights(self.modules)
        self.output_names = ["output_x", "output_y"]

    def forward(self, x):
        # N C H W -> N C 1 1
        x = self.avgpool(x)
        x = self.splitconcat(x)
        x = x.view(x.shape[0], -1)
        return {"output_x": self.fcx(x), "output_y": self.fcy(x)}

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}"


REGRESSOR_HEADS_REGISTRY.register(DummyMultipleHead)


def _get_base_regressor_config():
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = "FBNetV2C4Backbone"
    cfg.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
    cfg.MODEL.PIXEL_STD = [57.375, 57.12, 58.395]
    add_fbnet_v2_default_configs(cfg)
    add_quantization_default_configs(cfg)
    add_regressor_default_configs(cfg)
    cfg.MODEL.REGRESSOR.PREPROCESSORS.NAME = "SplitAndConcat"
    cfg.MODEL.FBNET_V2.STEM_IN_CHANNELS = 1
    return cfg


def _get_base_social_eye_config():
    cfg = _get_base_regressor_config()
    add_gaze_head_default_configs(cfg)
    add_smooth_l1_outlier_rejection_loss_default_configs(cfg)
    cfg.MODEL.REGRESSOR.LOSSES.NAME = "SmoothL1OutlierRejectionLoss"
    cfg.MODEL.REGRESSOR.HEAD_NAMES = ["GazeHead"]
    cfg.MODEL.REGRESSOR.TRAIN_HEAD_NAME = "GazeHead"
    return cfg


def _get_base_social_eye_with_multiple_config():
    # test with multiple heads and where 1 head has a dict output and the other a tensor
    cfg = _get_base_regressor_config()
    add_gaze_head_default_configs(cfg)
    add_smooth_l1_outlier_rejection_loss_default_configs(cfg)
    cfg.MODEL.REGRESSOR.LOSSES.NAME = "SmoothL1OutlierRejectionLoss"
    cfg.MODEL.REGRESSOR.HEAD_NAMES = ["DummyMultipleHead", "GazeHead"]
    cfg.MODEL.REGRESSOR.TRAIN_HEAD_NAME = "GazeHead"
    return cfg


class TestRegressor(unittest.TestCase):
    def _compare_dict_tensors(self, a, b, shape_only=False):
        """Compares the dict of tensors a, b"""
        self.assertEqual(len(a.keys()), len(b.keys()))
        for k, v_a in a.items():
            v_b = b[k]
            self.assertEqual(v_b.shape, v_a.shape)
            if not shape_only:
                torch.testing.assert_allclose(v_b, v_a)

    def test_init(self):
        """Check that the meta_arch can be built with a default config"""
        cfg = _get_base_social_eye_config()
        Regressor(cfg.clone())

    def test_inference(self):
        """Check that model inference runs"""
        cfg = _get_base_social_eye_config()
        model = Regressor(cfg.clone())

        # load dummy data
        batch_size = 16
        batch = [{"image": torch.randn(2, 32, 32)} for _ in range(batch_size)]
        model.eval()
        outputs = model(batch)

        # check output is the expected type and shape:
        #   should be list[dict], [{0: result0, 1: result1}]
        self.assertEqual(len(outputs), batch_size)
        self.assertTrue(isinstance(outputs[0]["gaze"], torch.Tensor))

    def test_forward_train(self):
        """Check that the forward pass during training includes loss function"""
        cfg = _get_base_social_eye_config()
        model = Regressor(cfg.clone())

        # load dummy data
        batch_size = 16
        batch = [
            # TODO: use more basic loss func
            {"image": torch.randn(2, 32, 32), "target": torch.randn(4)}
            for _ in range(batch_size)
        ]
        model.train()
        loss_dict = model(batch)

        # check output is the expected type and shape
        #   should be dict("loss0": value0, "loss1": value1)
        self.assertTrue(isinstance(loss_dict, dict))

    def test_deployable_model(self):
        """Check deployable model can use regressor"""
        cfg = _get_base_social_eye_config()
        model = Regressor(cfg.clone())
        model.eval()
        data = [{"image": torch.randn(2, 32, 32)}]
        tensor_input = [data[0]["image"].unsqueeze(0)]
        gt = model(data)[0]

        deployable_model = DeployableModel(model)
        output = deployable_model(tensor_input)
        for i, output_name in enumerate(model.get_ordered_head_output_names()):
            torch.testing.assert_allclose(output[i], gt[output_name].unsqueeze(0))

    def test_deployable_multiple_model(self):
        """Check deployable model can use regressor with multiple heads
        and where 1 head has a dict output and another head has a tensor output
        """
        cfg = _get_base_social_eye_with_multiple_config()
        model = Regressor(cfg.clone())
        model.eval()
        data = [{"image": torch.randn(2, 32, 32)}]
        tensor_input = [data[0]["image"].unsqueeze(0)]
        gt = model(data)[0]

        deployable_model = DeployableModel(model)
        output = deployable_model(tensor_input)
        for i, output_name in enumerate(model.get_ordered_head_output_names()):
            torch.testing.assert_allclose(output[i], gt[output_name].unsqueeze(0))

    def test_prepare_export(self):
        """Create predictor and compare with baseline model"""
        cfg = _get_base_social_eye_config()
        model = Regressor(cfg.clone())
        model.eval()
        data = [{"image": torch.randn(2, 32, 32)} for _ in range(5)]
        gts = model(data)

        # build predictor
        deployable_model = DeployableModel(model)
        export_config = model.prepare_for_export(cfg, [data], "caffe2")
        predictor = PredictorWrapper(
            model_or_models=deployable_model,
            run_func=export_config.run_func_info.instantiate(),
            preprocess=export_config.preprocess_info.instantiate(),
            postprocess=export_config.postprocess_info.instantiate(),
        )
        outputs = predictor(data)

        # compare the output of the predictor to the gt
        for gt, output in zip(gts, outputs):
            self.assertEqual(len(gt.keys()), len(output.keys()))
            for k, v in gt.items():
                # also make sure shape is the same
                self.assertEqual(output[k].shape, v.shape)
                torch.testing.assert_allclose(output[k], v)

    def test_prepare_multiple_export(self):
        """Create predictor and compare with baseline model with multiple heads
        and where 1 head has a dict output and another head has a tensor output
        """

        cfg = _get_base_social_eye_with_multiple_config()
        model = Regressor(cfg.clone())
        model.eval()
        data = [{"image": torch.randn(2, 32, 32)} for _ in range(5)]
        gts = model(data)

        # build predictor
        deployable_model = DeployableModel(model)
        export_config = model.prepare_for_export(cfg, [data], "caffe2")
        predictor = PredictorWrapper(
            model_or_models=deployable_model,
            run_func=export_config.run_func_info.instantiate(),
            preprocess=export_config.preprocess_info.instantiate(),
            postprocess=export_config.postprocess_info.instantiate(),
        )
        outputs = predictor(data)

        # compare the output of the predictor to the gt
        for gt, output in zip(gts, outputs):
            self.assertEqual(len(gt.keys()), len(output.keys()))
            for k, v in gt.items():
                # also make sure shape is the same
                self.assertEqual(output[k].shape, v.shape)
                torch.testing.assert_allclose(output[k], v)

    def test_prepare_for_quant(self):
        """Check running prepare_for_quant returns model that has quant
        ops and can be converted.

        Runs checks using prepare_for_quant, post_training_quantize,
        setup_qat_model
        """

        def _prepare_for_quant(cfg, model, data, eager_mode):
            model.eval()
            quant_model = model.prepare_for_quant(cfg)
            if eager_mode:
                torch.quantization.prepare(quant_model, inplace=True)
            output_quant = quant_model(data)
            return quant_model, output_quant

        def _d2go_ptq(cfg, model, data, eager_mode):
            quant_model = post_training_quantize(cfg, model, [data])
            output_quant = quant_model(data)
            return quant_model, output_quant

        def _d2go_qat(cfg, model, data, eager_mode):
            cfg = cfg.clone()
            cfg.QUANTIZATION.QAT.ENABLED = True
            model.train()
            quant_model = setup_qat_model(cfg, model, enable_observer=True)
            quant_model.eval()  # make sure bn is unchanged so we can compare the output
            output_quant = quant_model(data)
            return quant_model, output_quant

        qfuncs = [_prepare_for_quant, _d2go_ptq, _d2go_qat]
        # generate input data, all tests should be using the same intput data so that we
        # can compare the model
        input_data = [
            {"image": torch.randn(2, 32, 32), "target": torch.zeros(1)}
            for _ in range(5)
        ]
        # all models should use the same weights
        ref_cfg = _get_base_social_eye_config()
        ref_cfg.QUANTIZATION.EAGER_MODE = True
        ref_model = Regressor(ref_cfg.clone())

        for qfunc in qfuncs:
            # store the eager and graph results after prepare and convert
            # and compare them in the end
            results = {eager_mode : [] for eager_mode in [True, False]}
            for eager_mode in [True, False]:
                # create model
                cfg = _get_base_social_eye_config()
                cfg.QUANTIZATION.EAGER_MODE = eager_mode
                model = Regressor(cfg.clone())
                # load the model weight from ref model so that we start
                # with the models with same weights
                model.load_state_dict(ref_model.state_dict())
                model.eval()
                data = input_data[:]
                gts = model(data)
                results[eager_mode].append(gts)

                # run quantize
                quant_model, output_quant = qfunc(cfg, model, data, eager_mode)
                results[eager_mode].append(output_quant)

                if eager_mode:
                    # check we have quant dequant stubs
                    self.assertTrue(has_module(quant_model, QuantStub))
                    self.assertTrue(has_module(quant_model, DeQuantStub))

                # check that quant model is the same
                for a, b in zip(gts, output_quant):
                    self._compare_dict_tensors(a, b)

                # can the converted model run, check shapes
                if eager_mode:
                    converted_model = torch.quantization.convert(quant_model, inplace=False)
                else:
                    if hasattr(converted_model, "prepare_for_quant_convert"):
                        converted_model = quant_model.prepare_for_quant_convert(cfg)
                    else:
                        converted_model = torch.quantization.quantize_fx.convert_fx(quant_model)

                output_converted = converted_model(data)
                results[eager_mode].append(output_converted)
                for a, b in zip(gts, output_converted):
                    self._compare_dict_tensors(a, b, shape_only=True)

            # compare the results of eager mode quantized model
            # and fx graph mode quantized model
            for eager_res, graph_res in zip(results[True], results[False]):
                for e, g in zip(eager_res, graph_res):
                    self._compare_dict_tensors(e, g)

    def _test_run_export(self, cfg, predictor_type, qfunc=None):
        with make_temp_directory("test_export_predictor") as output_dir:
            model = Regressor(cfg.clone())
            model.eval()
            data = [{"image": torch.randn(2, 32, 32)} for _ in range(5)]
            gts = model(data)
            if qfunc is not None:
                model = qfunc(cfg.clone(), model, data)
            fname = convert_and_export_predictor(
                cfg.clone(), model, predictor_type, output_dir, [data]
            )
            saved_model = create_predictor(fname)
            output = saved_model(data)

            self.assertEqual(len(gts), len(output))
            for a, b in zip(gts, output):
                self._compare_dict_tensors(
                    a, b, shape_only="int8" in predictor_type
                )

    def test_run_export(self):
        """Check that model using predictor generates the same output

        Caffe2 fp32, torchscript fp32 should run
        """
        def _d2go_ptq(cfg, model, data):
            model.eval()
            return model

        def _d2go_qat(cfg, model, data):
            model.train()
            quant_model = setup_qat_model(cfg, model, enable_observer=True)
            quant_model.eval()  # make sure bn is unchanged so we can compare the output
            return quant_model

        options = itertools.product([True, False], [_d2go_ptq, _d2go_qat])
        for predictor_type in ["caffe2", "torchscript", "torchscript_int8"]:
            if "int8" in predictor_type:
                for eager_mode, qfunc in options:
                    cfg = _get_base_social_eye_with_multiple_config()
                    cfg.QUANTIZATION.EAGER_MODE = eager_mode
                    if qfunc is _d2go_qat:
                        cfg.QUANTIZATION.QAT.ENABLED = True
                    self._test_run_export(cfg.clone(), predictor_type, qfunc)
            else:
                cfg = _get_base_social_eye_with_multiple_config()
                self._test_run_export(cfg.clone(), predictor_type)


class TestSocialEyeRegressor(unittest.TestCase):
    def test_inference(self):
        """Check that model inference generates social eye output"""
        cfg = _get_base_social_eye_config()
        model = Regressor(cfg.clone())

        # load dummy data
        batch_size = 16
        batch = [{"image": torch.randn(2, 32, 32)} for _ in range(batch_size)]
        model.eval()
        outputs = model(batch)

        # check output is the expected type and shape:
        #   should be list[dict], [{0: result0, 1: result1}]
        self.assertEqual(len(outputs), batch_size)
        self.assertEqual(outputs[0]["gaze"].shape, torch.Size([4]))

    def test_forward_train(self):
        """Check that the forward pass during training returns smoothl1loss"""
        cfg = _get_base_social_eye_config()
        model = Regressor(cfg.clone())

        # load dummy data
        batch_size = 16
        batch = [
            {"image": torch.randn(2, 32, 32), "target": torch.randn(4)}
            for _ in range(batch_size)
        ]
        model.train()
        loss_dict = model(batch)

        # check output is the expected type and shape
        #   should be dict("loss0": value0, "loss1": value1)
        self.assertTrue("smoothl1ORloss" in loss_dict)
        self.assertEqual(loss_dict["smoothl1ORloss"].shape, torch.Size([]))


if __name__ == "__main__":
    unittest.main()
