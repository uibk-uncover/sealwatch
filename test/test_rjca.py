import conseal as cl
import jpeglib
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import tempfile
import unittest

from . import defs


class TestRJCA(unittest.TestCase):
    """Test suite for RJCA module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([
        [fname, alpha]
        for fname in defs.TEST_IMAGES
        for alpha in [.05, .1, .2, .4, .6, .8, 1.]
    ])
    def test_attack_stego(self, fname: str, alpha: float):
        self._logger.info(f'TestRJCA.test_attack_stego({fname}, {alpha})')
        # compress precover image
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        jpeglib.from_spatial(x[..., None]).write_spatial(self.tmp.name, qt=100)

        # load cover image
        jpeg = jpeglib.read_dct(self.tmp.name)

        # embed lsb replacement
        stego_dct = cl.lsb.simulate(
            jpeg.Y,
            alpha,
            seed=12345,
        )

        # estimate alpha with WS
        var_hat = sw.rjca.attack(stego_dct, jpeg.qt[jpeg.quant_tbl_no[0]])

        # test
        var_threshold = np.abs(var_hat - 1/12)
        self.assertLess(var_threshold, .015)

    @parameterized.expand([[fname] for fname in defs.TEST_IMAGES])
    def test_attack_cover(self, fname: str):
        self._logger.info(f'TestRJCA.test_attack_cover({fname})')
        # compress precover image
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        jpeglib.from_spatial(x[..., None]).write_spatial(self.tmp.name, qt=100)

        # load cover image
        jpeg = jpeglib.read_dct(self.tmp.name)

        # estimate alpha with WS
        var_hat = sw.rjca.attack(jpeg.Y, jpeg.qt[jpeg.quant_tbl_no[0]])

        # test
        var_threshold = np.abs(var_hat - 1/12)
        self.assertGreater(var_threshold, .015)
