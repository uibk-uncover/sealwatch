
import numpy as np
from parameterized import parameterized
import scipy.io
import sealwatch as sw
import unittest

# from sealwatch.features.spam import extract_spam686_features_from_file
# from sealwatch.utils.grouping import flatten_single
# from scipy.io import loadmat

from . import defs


FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'spam'


class TestSpam(unittest.TestCase):

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compare_matlab(self, fname):
        #
        path = defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg'
        f = sw.spam.extract_from_file(path)
        f = sw.tools.flatten_single(f)
        #
        path = FEATURES_DIR / f'{fname}.mat'
        f_ref = scipy.io.loadmat(path)["features"].flatten()
        np.testing.assert_allclose(f, f_ref)


__all__ = ["TestSpam"]
