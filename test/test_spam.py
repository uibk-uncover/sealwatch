import unittest
from parameterized import parameterized
from sealwatch.features.spam import extract_spam686_features_from_file
from sealwatch.utils.grouping import flatten_single
from scipy.io import loadmat
import numpy as np
from . import defs


FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'spam'


class TestSpam(unittest.TestCase):

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compare_matlab(self, fname):
        npy_spam_features = extract_spam686_features_from_file(
            defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
        npy_spam_features = flatten_single(npy_spam_features)
        matlab_spam_features = loadmat(
            FEATURES_DIR / f'{fname}.mat')["features"].flatten()

        np.testing.assert_allclose(npy_spam_features, matlab_spam_features)


__all__ = ["TestSpam"]
