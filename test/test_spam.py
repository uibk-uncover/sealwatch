import unittest
from parameterized import parameterized
from sealwatch.features.spam.spam import extract_spam686_features_from_filepath
from sealwatch.utils.grouping import flatten_single
from scipy.io import loadmat
import numpy as np
import os


BASE_DIR = "assets"


class TestSpam(unittest.TestCase):
    @parameterized.expand([
        ("cover/jpeg_75_gray/seal1.jpg", "features_matlab/spam/seal1.mat"),
        ("cover/jpeg_75_gray/seal2.jpg", "features_matlab/spam/seal2.mat"),
        ("cover/jpeg_75_gray/seal3.jpg", "features_matlab/spam/seal3.mat"),
        ("cover/jpeg_75_gray/seal4.jpg", "features_matlab/spam/seal4.mat"),
        ("cover/jpeg_75_gray/seal5.jpg", "features_matlab/spam/seal5.mat"),
        ("cover/jpeg_75_gray/seal6.jpg", "features_matlab/spam/seal6.mat"),
        ("cover/jpeg_75_gray/seal7.jpg", "features_matlab/spam/seal7.mat"),
        ("cover/jpeg_75_gray/seal8.jpg", "features_matlab/spam/seal8.mat"),
        ("cover/jpeg_75_gray/otter1.jpg", "features_matlab/spam/otter1.mat"),
        ("cover/jpeg_75_gray/otter2.jpg", "features_matlab/spam/otter2.mat"),
        ("cover/jpeg_75_gray/dolphin.jpg", "features_matlab/spam/dolphin.mat"),
    ])
    def test_compare_matlab(self, cover_filepath, matlab_features_filepath):
        npy_spam_features = extract_spam686_features_from_filepath(os.path.join(BASE_DIR, cover_filepath))
        npy_spam_features = flatten_single(npy_spam_features)
        matlab_spam_features = loadmat(os.path.join(BASE_DIR, matlab_features_filepath))["features"].flatten()

        np.testing.assert_allclose(npy_spam_features, matlab_spam_features)


__all__ = ["TestSpam"]
