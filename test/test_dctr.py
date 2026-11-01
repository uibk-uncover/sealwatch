
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
        """Default backend against the Matlab reference."""
        features = sw.dctr.extract_from_file(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg', qf=75)
        features = sw.tools.flatten(features)

        features_ref = scipy.io.loadmat(FEATURES_DIR / f'{fname}.mat')["features"].flatten()

        np.testing.assert_allclose(features, features_ref)


    @parameterized.expand([['seal1'], ['seal2'], ['odd'], ['int']])
    def test_separable(self, image):
        """Separable 1D convolutions give the same features as the 2D convolution."""
        rng = np.random.default_rng(12345)
        if image == 'odd':
            x = rng.normal(0, 30, (61, 45))
        elif image == 'int':
            x = rng.integers(-128, 128, (64, 64))
        else:
            x = sw.tools.jpeg.decompress_from_file(defs.COVER_COMPRESSED_GRAY_DIR / f'{image}.jpg') - 128
        with sw.BACKEND_PYTHON:
            f_separable = sw.dctr.extract(x, 4.0)
            f_2d = sw.dctr.extract(x, 4.0, separable=False)
        self.assertEqual(list(f_separable), list(f_2d))
        for k in f_separable:
            np.testing.assert_array_equal(f_separable[k], f_2d[k], err_msg=k)

    def test_compare_matlab_python(self):
        """Python implementation against the Matlab reference, on a single image."""
        with sw.BACKEND_PYTHON:
            features = sw.dctr.extract_from_file(defs.COVER_COMPRESSED_GRAY_DIR / 'seal1.jpg', qf=75)
        features_ref = scipy.io.loadmat(FEATURES_DIR / 'seal1.mat')["features"].flatten()
        np.testing.assert_allclose(sw.tools.flatten(features), features_ref)

    @parameterized.expand([
        ('seal1', 4.0), ('seal1', 1.0), ('seal1', 7.3),
        ('nonsquare', 4.0), ('odd', 4.0), ('int', 4.0),
    ])
    def test_rs_python(self, image, q):
        """Rust implementation is bit-identical to the Python implementation."""
        rng = np.random.default_rng(12345)
        x = sw.tools.jpeg.decompress_from_file(defs.COVER_COMPRESSED_GRAY_DIR / 'seal1.jpg') - 128
        x = {
            'seal1': x,
            'nonsquare': x[:, :300],
            'odd': rng.normal(0, 30, (61, 45)),
            'int': rng.integers(-128, 128, (64, 64)),
        }[image]
        with sw.BACKEND_PYTHON:
            f_py = sw.dctr.extract(x, q)
        with sw.BACKEND_RUST:
            f_rs = sw.dctr.extract(x, q)
        self.assertEqual(list(f_py), list(f_rs))
        for k in f_py:
            self.assertEqual(f_py[k].dtype, f_rs[k].dtype, k)
            np.testing.assert_array_equal(f_py[k], f_rs[k], err_msg=k)

    def test_rs_fallback(self):
        """Rust backend falls back to Python for the 2D convolution."""
        x = np.random.default_rng(12345).normal(0, 30, (40, 40))
        with sw.BACKEND_RUST:
            f_rs = sw.dctr.extract(x, 4.0, separable=False)
        with sw.BACKEND_PYTHON:
            f_py = sw.dctr.extract(x, 4.0, separable=False)
        for k in f_py:
            np.testing.assert_array_equal(f_rs[k], f_py[k], err_msg=k)

    def test_deprecated_x1(self):
        x = np.random.default_rng(12345).normal(0, 30, (40, 40))
        with self.assertWarns(DeprecationWarning):
            f_x1 = sw.dctr.extract(x1=x, q=4.0)
        f_x = sw.dctr.extract(x=x, q=4.0)
        for k in f_x:
            np.testing.assert_array_equal(f_x1[k], f_x[k])


__all__ = ["TestDctr"]
