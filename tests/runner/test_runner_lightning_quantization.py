#!/usr/bin/env python3

# pyre-unsafe
import os
import unittest
from unittest import mock

import torch
from d2go.runner.callbacks.quantization import (
    get_default_qat_qconfig,
    ModelTransform,
    PostTrainingQuantization,
    QuantizationAwareTraining,
    rgetattr,
    rhasattr,
    rsetattr,
)
from d2go.utils.misc import mode
from d2go.utils.testing.helper import tempdir
from d2go.utils.testing.lightning_test_module import TestModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.ao.quantization.qconfig import default_dynamic_qconfig, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx


class TestUtilities(unittest.TestCase):
    """Test some basic utilities we rely on."""

    def test_get_set_has(self):
        """Trivial test for generic behavior. Only support pre-existing deeply nested values."""

        class TestObject(object):
            def __init__(self):
                self.object = None
                self.set_to_five = 5

        obj = TestObject()
        obj.object = TestObject()
        obj.object.set_to_five = 10

        rsetattr(obj, "object.set_to_five", 1)
        self.assertTrue(rhasattr(obj, "object.set_to_five"))
        self.assertEqual(1, rgetattr(obj, "object.set_to_five"))
        self.assertEqual(5, rgetattr(obj, "set_to_five"))

        with self.assertRaises(AttributeError):
            rsetattr(obj, "object.does_not_exist.five", 5)


class TestModelTransform(unittest.TestCase):
    """Tests ModelTransforms."""

    def test_invalid_construction_type_error(self):
        """Validate construction of ModelTransforms. Always have fn, msg, and one of [step, interval]."""
        with self.assertRaises(TypeError):
            _ = ModelTransform()
        with self.assertRaises(TypeError):
            _ = ModelTransform(fn=lambda x: x)
        with self.assertRaises(TypeError):
            _ = ModelTransform(message="No function defined")
        with self.assertRaises(TypeError):
            _ = ModelTransform(
                fn=lambda x: x,
                message="Specified both step and interval",
                step=1,
                interval=1,
            )

    def test_positivity_value_error(self):
        """Validates ModelTransforms are constructed with only valid arguments."""

        def identity(x):
            return x

        with self.assertRaises(ValueError):
            _ = ModelTransform(fn=identity, message="Negative step", step=-1)
        with self.assertRaises(ValueError):
            _ = ModelTransform(fn=identity, message="Zero interval", interval=0)
        with self.assertRaises(ValueError):
            _ = ModelTransform(fn=identity, message="Negative interval", interval=-1)


@unittest.skip(
    "FX Graph Mode Quantization API has been updated, re-enable the test after PyTorch 1.13 stable release"
)
class TestQuantizationAwareTraining(unittest.TestCase):
    def test_qat_misconfiguration(self):
        """Tests failure when misconfiguring the QAT Callback."""
        invalid_params = [
            {"start_step": -1},
            {"enable_observer": (42, 42)},
            {"enable_observer": (42, 21)},
            {"enable_observer": (-1, None)},
            {"freeze_bn_step": -1},
        ]
        for invalid_param in invalid_params:
            with self.assertRaises(ValueError):
                _ = QuantizationAwareTraining(**invalid_param)

    def test_qat_transforms(self):
        """Tests the appropropriate ModelTransforms are defined with QAT."""
        qat = QuantizationAwareTraining(
            start_step=300, enable_observer=(350, 500), freeze_bn_step=550
        )

        trainer = Trainer()
        module = TestModule()

        qat.setup(trainer, module, stage="train")

        self.assertGreater(len(qat.transforms), 0)

        def assertContainsTransformsAtStep(step):
            """
            Asserts at least one transform exists at the specified step and
            that it is removed after the step begins.
            """
            self.assertGreater(
                len(
                    [
                        transform
                        for transform in qat.transforms
                        if transform.step == step
                    ]
                ),
                0,
                f"step={step}",
            )
            trainer.fit_loop.global_step = step
            qat.on_train_batch_start(trainer, module, batch=None, batch_idx=0)

            self.assertEqual(
                len(
                    [
                        transform
                        for transform in qat.transforms
                        if transform.step == step
                    ]
                ),
                0,
                f"step={step}",
            )

        assertContainsTransformsAtStep(step=300)
        assertContainsTransformsAtStep(step=350)
        assertContainsTransformsAtStep(step=500)
        assertContainsTransformsAtStep(step=550)

    @tempdir
    def test_qat_interval_transform(self, root_dir):
        """Tests an interval transform is applied multiple times."""
        seed_everything(100)

        def linear_fn_counter(mod):
            if isinstance(mod, torch.nn.Linear):
                linear_fn_counter.count += 1

        linear_fn_counter.count = 0

        model = TestModule()
        num_epochs = 2
        qat = QuantizationAwareTraining()
        qat.transforms.append(
            ModelTransform(fn=linear_fn_counter, message="Counter", interval=10)
        )
        trainer = Trainer(
            default_root_dir=os.path.join(root_dir, "quantized"),
            enable_checkpointing=False,
            callbacks=[qat],
            max_epochs=num_epochs,
            logger=False,
        )
        trainer.fit(model)

        # Model has 2 linear layers.
        self.assertEqual(linear_fn_counter.count, 2 * (trainer.global_step // 10 + 1))

    @tempdir
    def test_module_quantized_during_train(self, root_dir):
        """Validate quantized aware training works as expected."""
        seed_everything(100)

        model = TestModule()
        test_in = torch.randn(1, 32)
        before_train = model.eval()(test_in)
        num_epochs = 2
        qat = QuantizationAwareTraining()
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            default_root_dir=os.path.join(root_dir, "quantized"),
            enable_checkpointing=False,
            callbacks=[qat],
            max_epochs=num_epochs,
            logger=False,
        )
        trainer.fit(model)

        self.assertIsNotNone(qat.prepared)
        self.assertIsNotNone(qat.quantized)

        test_out = model.eval()(test_in)
        self.assertGreater(
            (test_out**2).sum(), 0.03, "With the given seend, L2^2 should be > 0.03."
        )

        base_out = qat.quantized.eval()(test_in)
        self.assertTrue(torch.allclose(base_out, test_out))
        # Weight changed during training.
        self.assertFalse(torch.allclose(before_train, test_out))

        # Validate .test() call works as expected and does not change model weights.
        trainer.test(model)

        self.assertTrue(torch.allclose(test_out, model.eval()(test_in)))

    @tempdir
    def test_quantization_without_train(self, root_dir):
        """Validate quantization occurs even without a call to .fit() first."""
        seed_everything(100)

        model = TestModule()
        num_epochs = 2
        qat = QuantizationAwareTraining()
        trainer = Trainer(
            default_root_dir=os.path.join(root_dir, "quantized"),
            enable_checkpointing=False,
            callbacks=[qat],
            max_epochs=num_epochs,
            logger=False,
        )

        trainer.test(model)

        self.assertIsNotNone(qat.prepared)
        self.assertIsNotNone(qat.quantized)

    @tempdir
    def test_attribute_preservation_qat(self, root_dir):
        """Validates we can preserve specified properties in module."""
        seed_everything(100)

        model = TestModule()
        model.layer._added_property = 10
        model._not_preserved = 15
        model._added_property = 20

        num_epochs = 2
        qat = QuantizationAwareTraining(
            preserved_attrs=["_added_property", "layer._added_property"]
        )
        trainer = Trainer(
            default_root_dir=os.path.join(root_dir, "quantized"),
            enable_checkpointing=False,
            callbacks=[qat],
            max_epochs=num_epochs,
            logger=False,
        )

        trainer.fit(model)

        self.assertIsNotNone(qat.prepared)
        self.assertIsNotNone(qat.quantized)

        # Assert properties are maintained.
        self.assertTrue(hasattr(qat.prepared, "_added_property"))
        self.assertTrue(hasattr(qat.prepared.layer, "_added_property"))

        with self.assertRaises(AttributeError):
            qat.prepared._not_preserved

    @tempdir
    def test_quantization_and_checkpointing(self, root_dir):
        """Validate written checkpoints can be loaded back as expected."""
        seed_everything(100)

        model = TestModule()
        num_epochs = 2
        qat = QuantizationAwareTraining()
        checkpoint_dir = os.path.join(root_dir, "checkpoints")
        checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, save_last=True)
        trainer = Trainer(
            default_root_dir=os.path.join(root_dir, "quantized"),
            callbacks=[qat, checkpoint],
            max_epochs=num_epochs,
            logger=False,
        )
        # Mimick failing mid-training by not running on_fit_end.
        with mock.patch.object(qat, "on_fit_end"):
            trainer.fit(model)

        ckpt = torch.load(os.path.join(checkpoint_dir, "last.ckpt"))
        model.load_state_dict(ckpt["state_dict"])

    @tempdir
    def test_custom_qat(self, root_dir):
        """Tests that we can customize QAT by skipping certain layers."""

        class _CustomQAT(QuantizationAwareTraining):
            """Only quantize TestModule.another_layer."""

            def prepare(self, model, configs, attrs):
                example_inputs = (torch.rand(1, 2),)
                model.another_layer = prepare_qat_fx(
                    model.another_layer, configs[""], example_inputs
                )

                return model

            def convert(self, model, submodules, attrs):
                model.another_layer = convert_fx(model.another_layer)
                return model

        seed_everything(100)
        model = TestModule()
        test_in = torch.randn(1, 32)
        before_train = model.eval()(test_in)
        num_epochs = 2
        qat = _CustomQAT()
        trainer = Trainer(
            default_root_dir=os.path.join(root_dir, "quantized"),
            enable_checkpointing=False,
            callbacks=[qat],
            max_epochs=num_epochs,
            logger=False,
        )
        trainer.fit(model)

        self.assertIsNotNone(qat.prepared)
        self.assertIsNotNone(qat.quantized)

        test_out = model.eval()(test_in)
        self.assertGreater(
            (test_out**2).sum(), 0.03, "With the given seend, L2^2 should be > 0.03."
        )

        base_out = qat.quantized.eval()(test_in)
        self.assertTrue(torch.allclose(base_out, test_out))
        # Weight changed during training.
        self.assertFalse(torch.allclose(before_train, test_out))

        # Validate .test() call works as expected and does not change model weights.
        trainer.test(model)

        self.assertTrue(torch.allclose(test_out, model.eval()(test_in)))

    @tempdir
    def test_submodule_qat(self, root_dir):
        """Tests that we can customize QAT through exposed API."""
        seed_everything(100)

        model = TestModule()
        test_in = torch.randn(1, 32)
        before_train = model.eval()(test_in)
        num_epochs = 2
        qat = QuantizationAwareTraining(
            qconfig_dicts={"another_layer": {"": get_default_qat_qconfig()}}
        )
        trainer = Trainer(
            default_root_dir=os.path.join(root_dir, "quantized"),
            enable_checkpointing=False,
            callbacks=[qat],
            max_epochs=num_epochs,
            logger=False,
        )
        trainer.fit(model)

        self.assertIsNotNone(qat.prepared)
        self.assertIsNotNone(qat.quantized)

        test_out = model.eval()(test_in)
        self.assertGreater(
            (test_out**2).sum(), 0.03, "With the given seend, L2^2 should be > 0.03."
        )

        base_out = qat.quantized.eval()(test_in)
        self.assertTrue(torch.allclose(base_out, test_out))
        # Weight changed during training.
        self.assertFalse(torch.allclose(before_train, test_out))

        # Validate .test() call works as expected and does not change model weights.
        trainer.test(model)

        self.assertTrue(torch.allclose(test_out, model.eval()(test_in)))


@unittest.skip(
    "FX Graph Mode Quantization API has been updated, re-enable the test after PyTorch 1.13 stable release"
)
class TestPostTrainingQuantization(unittest.TestCase):
    @tempdir
    def test_post_training_static_quantization(self, root_dir):
        """Validate post-training static quantization."""
        seed_everything(100)

        model = TestModule()
        num_epochs = 4
        static_quantization = PostTrainingQuantization(
            qconfig_dicts={"": {"": get_default_qconfig()}}
        )
        trainer = Trainer(
            default_root_dir=os.path.join(root_dir, "quantized"),
            enable_checkpointing=False,
            callbacks=[static_quantization],
            max_epochs=num_epochs,
            logger=False,
        )
        # This will both train the model + quantize it.
        trainer.fit(model)

        self.assertIsNotNone(static_quantization.quantized)
        # Default qconfig requires calibration.
        self.assertTrue(static_quantization.should_calibrate)

        test_in = torch.randn(12, 32)
        with mode(model, training=False) as m:
            base_out = m(test_in)
        with mode(static_quantization.quantized, training=False) as q:
            test_out = q(test_in)

        # While quantized/original won't be exact, they should be close.
        self.assertLess(
            ((((test_out - base_out) ** 2).sum(axis=1)) ** (1 / 2)).mean(),
            0.015,
            "RMSE should be less than 0.015 between quantized and original.",
        )

    @tempdir
    def test_post_training_dynamic_quantization(self, root_dir):
        """Validates post-training dynamic quantization."""
        seed_everything(100)

        model = TestModule()
        num_epochs = 2
        dynamic_quant = PostTrainingQuantization(
            qconfig_dicts={"": {"": default_dynamic_qconfig}}
        )
        trainer = Trainer(
            default_root_dir=os.path.join(root_dir, "quantized"),
            enable_checkpointing=False,
            callbacks=[dynamic_quant],
            max_epochs=num_epochs,
            logger=False,
        )
        # This will both train the model + quantize it.
        trainer.fit(model)

        self.assertIsNotNone(dynamic_quant.quantized)
        # Default qconfig requires calibration.
        self.assertFalse(dynamic_quant.should_calibrate)

        test_in = torch.randn(12, 32)
        with mode(model, training=False) as m:
            base_out = m(test_in)
        with mode(dynamic_quant.quantized, training=False) as q:
            test_out = q(test_in)

        # While quantized/original won't be exact, they should be close.
        self.assertLess(
            ((((test_out - base_out) ** 2).sum(axis=1)) ** (1 / 2)).mean(),
            0.015,
            "RMSE should be less than 0.015 between quantized and original.",
        )

    @tempdir
    def test_custom_post_training_static_quant(self, root_dir):
        """Tests that we can customize Post-Training static by skipping certain layers."""

        class _CustomStaticQuant(PostTrainingQuantization):
            """Only quantize TestModule.another_layer."""

            def prepare(self, model, configs, attrs):
                example_inputs = (torch.randn(1, 2),)
                model.another_layer = prepare_fx(
                    model.another_layer, configs[""], example_inputs
                )

                return model

            def convert(self, model, submodules, attrs):
                model.another_layer = convert_fx(model.another_layer)
                return model

        seed_everything(100)

        model = TestModule()
        num_epochs = 2
        static_quantization = _CustomStaticQuant()
        trainer = Trainer(
            default_root_dir=os.path.join(root_dir, "quantized"),
            enable_checkpointing=False,
            callbacks=[static_quantization],
            max_epochs=num_epochs,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(model)

        self.assertIsNotNone(static_quantization.quantized)

        test_in = torch.randn(12, 32)
        with mode(model, training=False) as m:
            base_out = m(test_in)
        with mode(static_quantization.quantized, training=False) as q:
            test_out = q(test_in)

        # While quantized/original won't be exact, they should be close.
        self.assertLess(
            ((((test_out - base_out) ** 2).sum(axis=1)) ** (1 / 2)).mean(),
            0.02,
            "RMSE should be less than 0.007 between quantized and original.",
        )
