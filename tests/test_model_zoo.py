import os
import unittest

from d2go.model_zoo import model_zoo

OSSRUN = os.getenv('OSSRUN') == '1'

class TestD2GoModelZoo(unittest.TestCase):
    @unittest.skipIf(not OSSRUN, "OSS test only")
    def test_model_zoo_pretrained(self):
        configs = list(model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys())
        for cfgfile in configs:
            model = model_zoo.get(cfgfile, trained=True)

if __name__ == "__main__":
    unittest.main()
