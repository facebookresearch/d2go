import os
import tempfile

from d2go.utils.flop_calculator import dump_flops_info
from d2go.utils.testing.data_loader_helper import (
    create_detection_data_loader_on_toy_dataset,
)
from d2go.utils.testing.rcnn_helper import RCNNBaseTestCases


class TestFlopCount(RCNNBaseTestCases.TemplateTestCase):
    def setup_custom_test(self):
        super().setup_custom_test()
        self.cfg.merge_from_file("detectron2go://mask_rcnn_fbnetv3a_dsmask_C4.yaml")

    def test_flop_count(self):
        size_divisibility = max(self.test_model.backbone.size_divisibility, 10)
        h, w = size_divisibility, size_divisibility * 2
        with create_detection_data_loader_on_toy_dataset(
            self.cfg, h, w, is_train=False
        ) as data_loader:
            inputs = (next(iter(data_loader)),)

        with tempfile.TemporaryDirectory(prefix="d2go_test") as output_dir:
            dump_flops_info(self.test_model, inputs, output_dir)

            for fname in [
                "flops_str_mobilecv",
                "flops_str_fvcore",
                "flops_table_fvcore",
            ]:
                outf = os.path.join(output_dir, fname + ".txt")
                self.assertTrue(os.path.isfile(outf))
