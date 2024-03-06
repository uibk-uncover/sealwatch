
import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import stegolab2 as sl2
import sys
import tempfile
import unittest

import defs


class TestChi2(unittest.TestCase):
    """Test suite for chi2 steganalysis module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[fname] for fname in defs.TEST_UG_IMAGES])
    def test_chi2_cover(self, fname: str):
        self._logger.info(f'TestChi2.test_chi2_cover({fname})')
        # load cover image
        x = np.array(Image.open(defs.COVER_UG_DIR / f'{fname}.png'))
        # run attack
        score, p_stego = sw.chi2.attack(x)
        # test
        self.assertLess(p_stego, .05)

    def run_lsbr_chi2(self, fname: str, alpha: float) -> float:
        # load cover image
        x = np.array(Image.open(defs.COVER_UG_DIR / f'{fname}.png'))
        x = x[..., None]
        # embed lsb
        y = cl.lsb.simulate(x, alpha, permute=True, seed=12345)
        # run chi2 histogram attack
        score, p_stego = sw.chi2.attack(y)
        return p_stego

    @parameterized.expand([
        [fname, alpha]
        for fname in defs.TEST_UG_IMAGES
        for alpha in [.95, 1.]
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
        for fname in defs.TEST_UG_IMAGES
        for alpha in [.1, .25]
    ])
    def test_chi2_stego_negative(self, fname: str, alpha: float):
        """"""
        self._logger.info(f'TestChi2.test_chi2_stego_negative({fname}, {alpha})')
        # print(fname, alpha, cond, thres)
        # simulate LSB and attack with chi2
        p_stego = self.run_lsbr_chi2(fname, alpha)

        # check result
        self.assertLess(p_stego, .01)

    # def embed_lsbr_path(self, x:np.ndarray, alpha:float, seed:int, generator:str=None) -> np.ndarray:
    #     # random message
    #     rng = np.random.default_rng(seed)
    #     message = rng.integers(0, 2, int(alpha * x.size))
    #     # select path
    #     perm = revelio.chi2.get_path(x, generator=generator, seed=seed)
    #     # embed
    #     y = x.copy()
    #     for m_it, (h, w, c) in enumerate(zip(*perm)):
    #         if len(message) <= m_it:
    #             break
    #         y[h, w, c] = (y[h, w, c] & ~0x1) | message[m_it]
    #     return y

    # @parameterized.expand([['lizard'], ['mountain'], ['nuclear']])
    # def test_chi2_path_cover(self, fname:str):
    #     """"""
    #     # load cover
    #     x = np.array(Image.open(f'img/{fname}.png'))
    #     # attack
    #     m_grid = np.arange(0, 1000, 100)
    #     scores, pvalues = revelio.chi2.attack_along_path(x, m_grid, seed=None)
    #     # estimate alpha
    #     alpha_hat = m_grid[np.argmax(pvalues < .95)] / x.size
    #     self.assertLess(alpha_hat, 1e-3)

    # @parameterized.expand([['lizard'], ['mountain'], ['nuclear']])
    # def test_chi2_path_sequential_stego(self, fname:str):
    #     """"""
    #     # load cover
    #     x = np.array(Image.open(f'img/{fname}.png'))
    #     # embed
    #     y = self.embed_lsbr_path(x, .4, seed=None)
    #     # attack
    #     m_grid = np.arange(0, y.size, 1000)
    #     scores, pvalues = revelio.chi2.attack_along_path(y, m_grid, seed=None)
    #     # estimate alpha
    #     alpha_hat = m_grid[np.argmax(pvalues < .95)] / x.size
    #     np.testing.assert_allclose(alpha_hat, .4, rtol=.2)  # quite imprecise estimate

    # @parameterized.expand([['mountain'], ['nuclear']])  # ['lizard'] # does not work for some reason
    # def test_chi2_path_permuted_correct_stego(self, fname:str):
    #     """"""
    #     # load cover
    #     x = np.array(Image.open(f'img/{fname}.png'))
    #     # embed
    #     y = self.embed_lsbr_path(x, .4, seed=12345)
    #     # attack
    #     m_grid = np.arange(int(y.size*.2), int(y.size*.6), 250)
    #     scores, pvalues = revelio.chi2.attack_along_path(y, m_grid, seed=12345)
    #     # estimate alpha
    #     alpha_hat = m_grid[np.argmax(pvalues < .95)] / x.size
    #     np.testing.assert_allclose(alpha_hat, .4, rtol=.2)  # quite imprecise estimate

    # @parameterized.expand([['lizard'], ['mountain'], ['nuclear']])
    # def test_chi2_path_permuted_wrong_stego(self, fname:str):
    #     """"""
    #     # load cover
    #     x = np.array(Image.open(f'img/{fname}.png'))
    #     # embed
    #     y = self.embed_lsbr_path(x, .4, seed=12345)
    #     # attack
    #     m_grid = np.arange(0, y.size, 1000)
    #     scores, pvalues = revelio.chi2.attack_along_path(y, m_grid, seed=54321)
    #     # estimate alpha
    #     alpha_hat = m_grid[np.argmax(pvalues < .95)] / x.size
    #     self.assertLess(alpha_hat, .2)  # underestimated (becase of wrong passwrod)
