import conseal as cl
import logging
import numpy as np
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import unittest
from . import defs


class TestMVG(unittest.TestCase):
    """Test suite for MVG module."""
    _logger = logging.getLogger(__name__)

    @staticmethod
    def _load(fname: str) -> np.ndarray:
        return np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))

    @staticmethod
    def _hill(x0: np.ndarray, alpha: float = .4) -> np.ndarray:
        probability = getattr(cl.simulate, 'probability', None) or cl.simulate._ternary.probability
        ps, _ = probability(cl.hill.compute_cost_adjusted(x0), alpha=alpha, n=x0.size)
        return np.asarray(ps)

    @parameterized.expand([[fname] for fname in defs.TEST_IMAGES[:2]])
    def test_variance_mipod(self, fname: str):
        self._logger.info(f'TestMVG.test_variance_mipod({fname})')
        x0 = self._load(fname).astype('float64')
        # MiPOD in conseal, with zero padding
        residual = x0 - cl.mipod._defs.wiener2(x0, kernel_size=(2, 2))
        v_ref = np.clip(cl.mipod.estimate_variance(residual, block_size=9, degree=9), .01, None)
        v = sw.mvg.estimate_variance(x0, boundary='fill')
        np.testing.assert_allclose(v, v_ref, rtol=1e-10)

    @parameterized.expand([[fname] for fname in defs.TEST_IMAGES[:2]])
    def test_taylor_exact(self, fname: str):
        self._logger.info(f'TestMVG.test_taylor_exact({fname})')
        x0 = self._load(fname)
        ps = self._hill(x0)  # asymmetric at saturated pixels
        d_taylor = sw.mvg.deflection(x0, ps)
        d_exact = sw.mvg.deflection(x0, ps, method='exact', nodes=20)
        np.testing.assert_allclose(d_exact, d_taylor, rtol=1e-2)

    @parameterized.expand([[fname] for fname in defs.TEST_IMAGES])
    def test_indifferent_below_omniscient(self, fname: str):
        self._logger.info(f'TestMVG.test_indifferent_below_omniscient({fname})')
        x0 = self._load(fname)
        ps = self._hill(x0)
        d_omniscient = sw.mvg.deflection(x0, ps)
        d_indifferent = sw.mvg.deflection(x0, ps, ps_hat='indifferent')
        self.assertLessEqual(d_indifferent, d_omniscient)

    @parameterized.expand([[fname] for fname in defs.TEST_IMAGES])
    def test_mipod_least_detectable(self, fname: str):
        self._logger.info(f'TestMVG.test_mipod_least_detectable({fname})')
        x0 = self._load(fname)
        ps_mipod = np.asarray(cl.mipod.probability(x0, alpha=.4)[0])
        d_mipod = sw.mvg.deflection(x0, ps_mipod, boundary='fill')
        d_hill = sw.mvg.deflection(x0, self._hill(x0), boundary='fill')
        self.assertLess(d_mipod, d_hill)


__all__ = ['TestMVG']
