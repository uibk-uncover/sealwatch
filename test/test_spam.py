import unittest
from parameterized import parameterized
from sealwatch.features.spam.spam import extract_spam686_features_from_filepath
from sealwatch.utils.grouping import flatten_single
from scipy.io import loadmat
import numpy as np
import os


BASE_DIR = "tests/assets"


class TestSpam(unittest.TestCase):
    @parameterized.expand([
        ("cover/00001.jpeg", "features_matlab/00001_spam.mat"),
        ("cover/00002.jpeg", "features_matlab/00002_spam.mat"),
        ("cover/00003.jpeg", "features_matlab/00003_spam.mat"),
        ("cover/00004.jpeg", "features_matlab/00004_spam.mat"),
        ("cover/00005.jpeg", "features_matlab/00005_spam.mat"),
    ])
    def test_compare_matlab(self, cover_filepath, matlab_features_filepath):
        npy_spam_features = extract_spam686_features_from_filepath(os.path.join(BASE_DIR, cover_filepath))
        npy_spam_features = flatten_single(npy_spam_features)
        matlab_spam_features = loadmat(os.path.join(BASE_DIR, matlab_features_filepath))["features"].flatten()

        np.testing.assert_allclose(npy_spam_features, matlab_spam_features)


__all__ = ["TestSpam"]
