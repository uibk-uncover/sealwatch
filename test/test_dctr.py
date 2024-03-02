import unittest
from parameterized import parameterized
from sealwatch.features.dctr.dctr import extract_dctr_features
from scipy.io import loadmat
import numpy as np
import os


ASSETS_DIR = "assets"


class TestDctr(unittest.TestCase):
    @parameterized.expand([
        ("cover/jpeg_75_gray/dolphin.jpg", "features_matlab/dctr/dolphin.mat"),
        ("cover/jpeg_75_gray/hafelekar.jpg", "features_matlab/dctr/hafelekar.mat"),
        ("cover/jpeg_75_gray/otter1.jpg", "features_matlab/dctr/otter1.mat"),
        ("cover/jpeg_75_gray/otter2.jpg", "features_matlab/dctr/otter2.mat"),
        ("cover/jpeg_75_gray/patscherkofel1.jpg", "features_matlab/dctr/patscherkofel1.mat"),
        ("cover/jpeg_75_gray/patscherkofel2.jpg", "features_matlab/dctr/patscherkofel2.mat"),
        ("cover/jpeg_75_gray/seal1.jpg", "features_matlab/dctr/seal1.mat"),
        ("cover/jpeg_75_gray/seal2.jpg", "features_matlab/dctr/seal2.mat"),
        ("cover/jpeg_75_gray/seal3.jpg", "features_matlab/dctr/seal3.mat"),
        ("cover/jpeg_75_gray/seal4.jpg", "features_matlab/dctr/seal4.mat"),
    ])
    def test_compare_matlab(self, cover_filepath, matlab_features_filepath):
        cover_filepath = os.path.join(ASSETS_DIR, cover_filepath)
        dctr_features = extract_dctr_features(cover_filepath, qf=75)
        dctr_features = dctr_features.flatten()

        matlab_features_filepath = os.path.join(ASSETS_DIR, matlab_features_filepath)
        gfr_features_matlab = loadmat(matlab_features_filepath)["features"].flatten()

        np.testing.assert_allclose(dctr_features, gfr_features_matlab)


__all__ = ["TestDctr"]
