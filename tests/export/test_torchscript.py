import unittest

from d2go.export.torchscript import (
    MobileOptimizationConfig,
    update_export_kwargs_from_export_method,
)


@update_export_kwargs_from_export_method
def mock_export(cls, model, input_args, save_path, export_method, **export_kwargs):
    # Return the export kwargs, so that we can check to make sure it's set as expected
    return export_kwargs


class TestTorchscriptExportMethods(unittest.TestCase):
    def test_update_export_kwargs_from_export_method(self):

        _empty_export_kwargs = {}

        def try_mock_export(export_method: str, export_kwargs=_empty_export_kwargs):
            return mock_export(
                cls=None,
                model=None,
                input_args=None,
                save_path=None,
                export_method=export_method,
                **export_kwargs,
            )

        export_method_string = "torchscript"
        new_export_kwargs = try_mock_export(export_method_string)
        self.assertNotIn("mobile_optimization", new_export_kwargs)

        export_method_string = "torchscript_mobile"
        new_export_kwargs = try_mock_export(export_method_string)
        self.assertIn("mobile_optimization", new_export_kwargs)
        self.assertEqual(
            type(new_export_kwargs["mobile_optimization"]),
            MobileOptimizationConfig,
        )
        self.assertEqual(new_export_kwargs["mobile_optimization"].backend, "CPU")

        export_method_string = "torchscript_mobile-metal"
        new_export_kwargs = try_mock_export(export_method_string)
        self.assertEqual(new_export_kwargs["mobile_optimization"].backend, "metal")

        export_method_string = "torchscript_mobile-vulkan"
        new_export_kwargs = try_mock_export(export_method_string)
        self.assertEqual(new_export_kwargs["mobile_optimization"].backend, "vulkan")

        export_method_string = "torchscript_mobile@tracing"
        new_export_kwargs = try_mock_export(export_method_string)
        self.assertEqual(new_export_kwargs["jit_mode"], "trace")

        export_method_string = "torchscript_mobile@scripting"
        new_export_kwargs = try_mock_export(export_method_string)
        self.assertEqual(new_export_kwargs["jit_mode"], "script")
