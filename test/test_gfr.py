import unittest
from parameterized import parameterized
from sealwatch.features.gfr.gfr import extract_gfr_features_from_file as extract_gfr_original_features_from_file
from scipy.io import loadmat
import numpy as np
import os


BASE_DIR = "tests/assets"


class TestGfr(unittest.TestCase):
    @parameterized.expand([
        ("cover/00001.jpeg", "features_matlab/00001_gfr.mat"),
        ("cover/00002.jpeg", "features_matlab/00002_gfr.mat"),
        ("cover/00003.jpeg", "features_matlab/00003_gfr.mat"),
        ("cover/00004.jpeg", "features_matlab/00004_gfr.mat"),
        ("cover/00005.jpeg", "features_matlab/00005_gfr.mat"),
    ])
    def test_compare_matlab(self, cover_filepath, matlab_features_filepath):
        gfr_features_npy = extract_gfr_original_features_from_file(os.path.join(BASE_DIR, cover_filepath), num_rotations=32, qf=75)
        gfr_features_npy = gfr_features_npy.flatten()
        gfr_features_matlab = loadmat(os.path.join(BASE_DIR, matlab_features_filepath))["features"].flatten()

        # TODO: Reduce atol
        np.testing.assert_allclose(gfr_features_npy, gfr_features_matlab, atol=1e-6)


__all__ = ["TestGfr"]
