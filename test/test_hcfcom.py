
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import sys
import tempfile
import time
import unittest

sys.path.append('test')
import defs


class TestHCFCOM(unittest.TestCase):
    """Test suite for hcfcom module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_extract(self, fname):
        self._logger.info(f'TestHCFCOM.test_extract({fname})')

        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))

        # extract HCF-COM
        start = time.perf_counter()
        features = sw.hcfcom.extract(x0)
        features = sw.tools.flatten(features)
        end = time.perf_counter()
        self._logger.info(f'extracting HCF-COM from {fname}.tif took {end - start} s')

        #
        self.assertEqual(features.shape, (3,))
        # print(fname, features)
        # TODO: test
