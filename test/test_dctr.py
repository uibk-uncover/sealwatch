import unittest
from parameterized import parameterized
from sealwatch.features.dctr.dctr import extract_dctr_features
from scipy.io import loadmat
import numpy as np
import os

from . import defs
FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'dctr'

# ASSETS_DIR = "assets"


class TestDctr(unittest.TestCase):

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compare_matlab(self, fname):
        dctr_features = extract_dctr_features(defs.COVER_CG_DIR / f'{fname}.jpg', qf=75)
        dctr_features = dctr_features.flatten()

        # matlab_features_filepath = os.path.join(ASSETS_DIR, matlab_features_filepath)
        dctr_features_matlab = loadmat(FEATURES_DIR / f'{fname}.mat')["features"].flatten()

        np.testing.assert_allclose(dctr_features, dctr_features_matlab)


__all__ = ["TestDctr"]
