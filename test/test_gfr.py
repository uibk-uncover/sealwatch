import numpy as np
from parameterized import parameterized
import scipy.io
import sealwatch as sw
import unittest

from . import defs


FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'gfr'


class TestGfr(unittest.TestCase):

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compare_matlab(self, fname):
        """Default backend against the Matlab reference."""
        features = sw.gfr.extract_from_file(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg', num_rotations=32, qf=75)
        self.assertEqual(len(features), 2 * 4 * 17 * 25)
        features = sw.tools.flatten(features)

        features_ref = scipy.io.loadmat(FEATURES_DIR / f'{fname}.mat')["features"].flatten()

        np.testing.assert_allclose(features, features_ref, atol=1e-6)

    def test_compare_matlab_python(self):
        """Python implementation against the Matlab reference, on a single image."""
        with sw.BACKEND_PYTHON:
            features = sw.gfr.extract_from_file(defs.COVER_COMPRESSED_GRAY_DIR / 'seal1.jpg', num_rotations=32, qf=75)
        features_ref = scipy.io.loadmat(FEATURES_DIR / 'seal1.mat')["features"].flatten()
        np.testing.assert_allclose(sw.tools.flatten(features), features_ref, atol=1e-6)

    @parameterized.expand([
        ('seal1', {}),
        ('seal1', {'quantization_steps': [0.1, 1, 1.5, 2]}),  # qf 95
        ('seal1', {'implementation': sw.GFR_FIX}),
        ('seal1', {'num_rotations': 8, 'T': 2}),
        ('nonsquare', {}),
        ('odd', {}),
        ('int', {}),
    ])
    def test_rs_python(self, image, kw):
        """Rust implementation is bit-identical to the Python implementation."""
        rng = np.random.default_rng(12345)
        x = sw.tools.jpeg.decompress_from_file(defs.COVER_COMPRESSED_GRAY_DIR / 'seal1.jpg')
        x = {
            'seal1': x,
            'nonsquare': x[:300, ::2],  # not contiguous
            'odd': rng.normal(128, 30, (61, 45)),
            'int': rng.integers(0, 256, (64, 64), dtype='uint8'),
        }[image]
        with sw.BACKEND_PYTHON:
            f_py = sw.gfr.extract(x, **kw)
        with sw.BACKEND_RUST:
            f_rs = sw.gfr.extract(x, **kw)
        self.assertEqual(list(f_py), list(f_rs))
        for k in f_py:
            self.assertEqual(f_py[k].dtype, f_rs[k].dtype, k)
            np.testing.assert_array_equal(f_py[k], f_rs[k], err_msg=k)

    def test_group(self):
        """A flat feature vector is split into the submodels."""
        x = np.random.default_rng(12345).integers(0, 256, (64, 64), dtype='uint8')
        features = sw.gfr.extract(x)
        groups = sw.tools.features.group(sw.tools.flatten(features), sw.tools.features.Features.FEATURES_GFR)
        self.assertEqual(list(groups), list(features))
        for k in features:
            np.testing.assert_array_equal(groups[k], features[k])

    def test_invalid(self):
        x = np.zeros((16, 16))
        with self.assertRaises(ValueError):
            sw.gfr.extract(x, quantization_steps=[2, 4, 6])
        with self.assertRaises(ValueError):
            sw.gfr.extract(x, num_rotations=7)

    def test_deprecated_img(self):
        x = np.random.default_rng(12345).normal(128, 30, (40, 40))
        with self.assertWarns(DeprecationWarning):
            f_img = sw.gfr.extract(img=x)
        f_x = sw.gfr.extract(x=x)
        for k in f_x:
            np.testing.assert_array_equal(f_img[k], f_x[k])


__all__ = ["TestGfr"]
