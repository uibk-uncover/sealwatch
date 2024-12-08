
import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import sys
import tempfile
import unittest

from . import defs


class TestChi2(unittest.TestCase):
    """Test suite for chi2 steganalysis module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([
        [fname]
        for fname in defs.TEST_IMAGES
        if fname not in {'seal4', 'seal5'}  # avoid, raise FP
    ])
    def test_chi2_cover(self, fname: str):
        self._logger.info(f'TestChi2.test_chi2_cover({fname})')
        # load cover image
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        # run attack
        score, p_stego = sw.chi2.attack(x)
        # test
        self.assertLess(p_stego, .05)

    def run_lsbr_chi2(self, fname: str, alpha: float) -> float:
        # load cover image
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        x = x[..., None]
        # embed lsb
        y = cl.lsb.simulate(x, alpha, permute=True, seed=12345)
        # run chi2 histogram attack
        score, p_stego = sw.chi2.attack(y)
        return p_stego

    @parameterized.expand([
        [fname, alpha]
        for fname in defs.TEST_IMAGES
        for alpha in [.95, 1.]
        if fname not in {'seal4', 'seal5'}  # avoid, raise FP
    ])
    def test_chi2_stego_positive(self, fname: str, alpha: float):
        """"""
        self._logger.info(f'TestChi2.test_chi2_stego_positive({fname}, {alpha})')
        # print(fname, alpha, cond, thres)
        # simulate LSB and attack with chi2
        p_stego = self.run_lsbr_chi2(fname, alpha)

        # check result
        self.assertGreater(p_stego, .99)

    @parameterized.expand([
        [fname, alpha]
        for fname in defs.TEST_IMAGES
        for alpha in [.1, .25]
        if fname not in {'seal4', 'seal5'}  # avoid, raise FP
    ])
    def test_chi2_stego_negative(self, fname: str, alpha: float):
        """"""
        self._logger.info(f'TestChi2.test_chi2_stego_negative({fname}, {alpha})')
        # print(fname, alpha, cond, thres)
        # simulate LSB and attack with chi2
        p_stego = self.run_lsbr_chi2(fname, alpha)

        # check result
        self.assertLess(p_stego, .01)
