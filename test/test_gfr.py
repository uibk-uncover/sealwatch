import unittest
from parameterized import parameterized
from sealwatch.features.gfr.gfr import extract_gfr_features_from_file as extract_gfr_original_features_from_file
from scipy.io import loadmat
import numpy as np
import os

from . import defs
FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'gfr'


ASSETS_DIR = "assets"


class TestGfr(unittest.TestCase):
    # @parameterized.expand([
    #     ("cover/jpeg_75_gray/seal1.jpg", "features_matlab/gfr/seal1.mat"),
    #     ("cover/jpeg_75_gray/seal2.jpg", "features_matlab/gfr/seal2.mat"),
    #     ("cover/jpeg_75_gray/seal3.jpg", "features_matlab/gfr/seal3.mat"),
    #     ("cover/jpeg_75_gray/seal4.jpg", "features_matlab/gfr/seal4.mat"),
    #     ("cover/jpeg_75_gray/seal5.jpg", "features_matlab/gfr/seal5.mat"),
    #     ("cover/jpeg_75_gray/seal6.jpg", "features_matlab/gfr/seal6.mat"),
    #     ("cover/jpeg_75_gray/seal7.jpg", "features_matlab/gfr/seal7.mat"),
    #     ("cover/jpeg_75_gray/seal8.jpg", "features_matlab/gfr/seal8.mat"),
    #     ("cover/jpeg_75_gray/otter1.jpg", "features_matlab/gfr/otter1.mat"),
    #     ("cover/jpeg_75_gray/otter2.jpg", "features_matlab/gfr/otter2.mat"),
    #     ("cover/jpeg_75_gray/dolphin.jpg", "features_matlab/gfr/dolphin.mat"),
    # ])
    # def test_compare_matlab(self, cover_filepath, matlab_features_filepath):
    #     gfr_features_npy = extract_gfr_original_features_from_file(os.path.join(ASSETS_DIR, cover_filepath), num_rotations=32, qf=75)
    #     gfr_features_npy = gfr_features_npy.flatten()
    #     gfr_features_matlab = loadmat(os.path.join(ASSETS_DIR, matlab_features_filepath))["features"].flatten()

    #     np.testing.assert_allclose(gfr_features_npy, gfr_features_matlab, atol=1e-6)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compare_matlab(self, fname):
        gfr_features_npy = extract_gfr_original_features_from_file(
            defs.COVER_CG_DIR / f'{fname}.jpg',
            num_rotations=32, qf=75,
        )
        gfr_features_npy = gfr_features_npy.flatten()
        gfr_features_matlab = loadmat(FEATURES_DIR / f'{fname}.mat')["features"].flatten()

        np.testing.assert_allclose(gfr_features_npy, gfr_features_matlab, atol=1e-6)




__all__ = ["TestGfr"]
