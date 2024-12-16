
import numpy as np
from parameterized import parameterized
import scipy.io
import sealwatch as sw
# from sealwatch.features.dctr import extract_dctr_features_from_file
import unittest

from . import defs


FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'dctr'


class TestDctr(unittest.TestCase):

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compare_matlab(self, fname):
        features = sw.dctr.extract_from_file(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg', qf=75)
        features = sw.tools.flatten(features)

        features_ref = scipy.io.loadmat(FEATURES_DIR / f'{fname}.mat')["features"].flatten()

        np.testing.assert_allclose(features, features_ref)


__all__ = ["TestDctr"]
