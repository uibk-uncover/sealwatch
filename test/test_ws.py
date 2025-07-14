
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


class TestWS(unittest.TestCase):
    """Test suite for WS module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[fname] for fname in defs.TEST_IMAGES])
    def test_attack_cover(self, fname: str):
        self._logger.info(f'TestWS.test_attack_cover({fname})')
        # load cover image
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        x = x[..., None]

        # estimate alpha with WS
        beta_hat = sw.ws.attack(x)

        # test
        np.testing.assert_allclose(beta_hat, 0, atol=.05)

    @parameterized.expand([
        [fname, alpha]
        for fname in defs.TEST_IMAGES
        for alpha in [.4]  # .05, .1, .2, .4, .6, .8, 1.]
    ])
    def test_attack_stego(self, fname: str, alpha: float):
        self._logger.info(f'TestWS.test_attack_stego({fname}, {alpha})')
        # load cover image
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        x = x[..., None]

        # embed lsb replacement
        y = cl.lsb.simulate(x, alpha, modify=cl.LSB_REPLACEMENT, permute=True, seed=12345)

        # estimate alpha with WS
        beta_hat = sw.ws.attack(y)

        # test
        np.testing.assert_allclose(beta_hat, alpha/2, atol=.1)

    def test_unet_estimator(self):
        self._logger.info('TestWS.test_unet_estimator()')
        #
        estimator = sw.ws.unet_estimator()
