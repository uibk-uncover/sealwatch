
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


import scipy.signal

def compute_cost(
    cover_spatial: np.ndarray,
) -> np.ndarray:
    # high-pass filter
    H_KB = np.array([
        [-1,  2, -1],
        [ 2, -4,  2],
        [-1,  2, -1]
    ], dtype='float32')
    I1 = scipy.signal.convolve2d(
        cover_spatial, H_KB,
        mode='same', boundary='symm',
    )

    # low-pass filter 1
    I2 = np.abs(I1)
    # L1 = np.ones((3, 3), dtype='float32') / 3**2
    # I2 = scipy.signal.convolve2d(
    #     I2, L1,
    #     mode='same', boundary='symm',
    # )

    # low-pass filter 2
    I2[I2 < cl.tools.EPS] = cl.tools.EPS
    I3 = 1./(I2+1e-15)
    L2 = np.ones((15, 15), dtype='float32')/15**2
    I3 = scipy.signal.convolve2d(
        I3, L2,
        mode='same', boundary='symm',
    )

    #
    return I3

class TestLRT(unittest.TestCase):
    """Test suite for lrt steganalysis module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[fname] for fname in ['seal1']])  # defs.TEST_IMAGES])
    def test_lrt(self, fname: str):
        self._logger.info(f'TestLRT.test_lrt({fname})')
        # load cover image
        x = np.array(Image.open(defs.COVER_UG_DIR / f'{fname}.png'))
        x = x.astype('float')
        # cover variance
        # sigma2 = sw.lrt.local_variance(x)
        # change rates
        # rho = cl.hill._costmap.compute_cost(x)
        rho = compute_cost(x)
        import stegolab2 as sl2
        (beta,), _ = sl2.simulate._binary.probability([rho], .8, x.size)
        # beta = np.ones(x.shape)
        # beta = None
        # LRT
        lbda = sw.lrt.attack(x, beta=beta)
        print(np.nansum(lbda))
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(x, cmap='gray')
        # ax[1].imshow(np.log10(lbda), cmap='gray')
        # plt.show()
        # # test
        # self.assertLess(p_stego, .05)