import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import tempfile
import unittest

from . import defs


class TestSPA(unittest.TestCase):
    """Test suite for SPA module."""
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
        if fname not in {'seal8'}  # estimate off
    ])
    def test_spa_cover(self, fname):
        self._logger.info(f'TestSPA.test_spa_cover({fname})')
        # load image
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))

        # run attack
        q = sw.spa.attack(x)

        # test
        self.assertLess(q, .1)

    @parameterized.expand([
        [fname, alpha]
        for fname in defs.TEST_IMAGES
        for alpha in [.25, .5, .75, 1.]
        if fname not in {'seal5', 'seal8'}  # estimate off
    ])
    def test_spa_stego(self, fname: str, alpha: float):
        self._logger.info(f'TestSPA.test_spa_stego({fname})')
        # load cover image
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))

        # embed lsb
        y = cl.lsb.simulate(x, alpha, seed=12345)

        # run SPA
        alpha_hat = sw.spa.attack(y)

        # test
        self.assertLess(abs(alpha_hat - alpha), .1)
