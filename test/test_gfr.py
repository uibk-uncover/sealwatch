import unittest
from parameterized import parameterized
import scipy.io
import numpy as np
import sealwatch as sw

# from sealwatch.features.gfr.gfr import extract_gfr_features_from_file as extract_gfr_original_features_from_file
from . import defs


FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'gfr'


class TestGfr(unittest.TestCase):

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compare_matlab(self, fname):
        gfr_features_npy = sw.gfr.extract_from_file(
            defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg',
            num_rotations=32, qf=75,
        )
        gfr_features_npy = gfr_features_npy.flatten()
        gfr_features_matlab = scipy.io.loadmat(FEATURES_DIR / f'{fname}.mat')["features"].flatten()

        np.testing.assert_allclose(gfr_features_npy, gfr_features_matlab, atol=1e-6)


__all__ = ["TestGfr"]
