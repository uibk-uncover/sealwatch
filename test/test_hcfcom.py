
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

    def test_constant(self):
        """Constant image has a flat HCF, |H[k]| = 1, so that the COM is the mean of k^order, k = 0, ..., 128."""
        x = np.full((32, 32), 100, dtype='uint8')
        self.assertAlmostEqual(sw.hcfcom.extract(x)['0'], 64)
        self.assertAlmostEqual(sw.hcfcom.extract(x, order=2)['0'], 128 * 257 / 6)

    def test_definition(self):
        """COM by the definition of Ker (2005), on a real image."""
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / 'seal1.png'))
        h = np.bincount(x.flatten(), minlength=256)
        H = np.abs([np.sum(h * np.exp(-2j * np.pi * k * np.arange(256) / 256)) for k in range(129)])
        for order in [1, 2, 3]:
            expected = sum(k ** order * H[k] for k in range(129)) / sum(H)
            self.assertAlmostEqual(sw.hcfcom.extract(x, order=order)['0'] / expected, 1, places=10)

    def test_deprecated_x1(self):
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / 'seal1.png'))
        with self.assertWarns(DeprecationWarning):
            f_x1 = sw.hcfcom.extract(x1=x)
        self.assertEqual(f_x1['0'], sw.hcfcom.extract(x=x)['0'])
