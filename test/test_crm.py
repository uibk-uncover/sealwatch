
from collections import OrderedDict
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import scipy.io
import sealwatch as sw
import sys
import tempfile
import time
import warnings
import unittest

sys.path.append('test')
import defs


FEATURE_DIR = defs.ASSETS_DIR / 'features_matlab' / 'crm'


class TestCRM(unittest.TestCase):
    """Test suite for CRM module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_extract(self, fname):
        """Default backend (CRM_FIX_MIN24) against the Matlab reference (CRM_ORIGINAL)."""
        self._logger.info(f'TestCRM.test_extract({fname})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))
        features = sw.crm.extract(x0, Tc=3)
        self.assertEqual(len(sw.tools.flatten(features)), 5404)  # from DDE page
        features_ref = scipy.io.loadmat(FEATURE_DIR / f'{fname}.mat', simplify_cells=True)
        features_ref = {k: v for k, v in features_ref.items() if not k.startswith('_')}
        self.assertEqual(set(features_ref), set(features))
        for k in features:
            if k == 's3_minmax24c_q1':
                # CRM_FIX_MIN24 counts it once, Matlab twice
                np.testing.assert_array_equal(2 * features[k], features_ref[k])
            else:
                np.testing.assert_array_equal(features[k], features_ref[k])
        features = sw.crm.extract(x0, Tc=2)
        self.assertEqual(len(sw.tools.flatten(features)), 2073)

    @parameterized.expand([['seal1'], ['seal2']])
    def test_extract_python(self, fname):
        """Python implementation (CRM_ORIGINAL) against the Matlab reference."""
        self._logger.info(f'TestCRM.test_extract_python({fname})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))
        with sw.BACKEND_PYTHON:
            features = sw.crm.extract(x0, Tc=3, implementation=sw.CRM_ORIGINAL)
        self.assertEqual(len(sw.tools.flatten(features)), 5404)
        features_ref = scipy.io.loadmat(FEATURE_DIR / f'{fname}.mat', simplify_cells=True)
        features_ref = {k: v for k, v in features_ref.items() if not k.startswith('_')}
        self.assertEqual(set(features_ref), set(features))
        for k in features:
            np.testing.assert_array_equal(features[k], features_ref[k])
        with sw.BACKEND_PYTHON:
            features = sw.crm.extract(x0, Tc=2, implementation=sw.CRM_ORIGINAL)
        self.assertEqual(len(sw.tools.flatten(features)), 2073)

    @parameterized.expand([[2], [3]])
    def test_implementation(self, Tc):
        """CRM_ORIGINAL doubles the 3rd-order minmax24, otherwise it equals CRM_FIX_MIN24."""
        self._logger.info(f'TestCRM.test_implementation({Tc=})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / 'seal1.png'))
        # only the Python implementation has CRM_ORIGINAL
        with sw.BACKEND_PYTHON:
            f_fix = sw.crm.extract(x0, Tc=Tc)
            f_orig = sw.crm.extract(x0, Tc=Tc, implementation=sw.CRM_ORIGINAL)
        self.assertEqual(list(f_fix), list(f_orig))
        for k in f_fix:
            if k == 's3_minmax24c_q1':
                np.testing.assert_array_equal(2 * f_fix[k], f_orig[k])
            else:
                np.testing.assert_array_equal(f_fix[k], f_orig[k])

    @parameterized.expand([
        ('seal1', {}),
        ('seal1', {'Tc': 3}),
        ('seal1', {'q': 1.5}),  # rounding in float32
        ('seal1', {'q': 2.0, 'Tc': 3}),
        ('odd', {}),
        ('flipped', {}),  # not contiguous
    ])
    def test_rs_python(self, image, kw):
        """Rust implementation is bit-identical to the Python implementation."""
        self._logger.info(f'TestCRM.test_rs_python({image=}, {kw=})')
        rng = np.random.default_rng(12345)
        if image == 'odd':
            x = rng.integers(0, 256, (37, 53, 3)).astype(np.uint8)
        else:
            x = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / 'seal1.png'))
            if image == 'flipped':
                x = x[::-1, ::2]
        with sw.BACKEND_PYTHON:
            f_py = sw.crm.extract(x, **kw)
        with sw.BACKEND_RUST:
            f_rs = sw.crm.extract(x, **kw)
        self.assertEqual(list(f_py.keys()), list(f_rs.keys()))
        for name in f_py:
            self.assertEqual(f_py[name].dtype, f_rs[name].dtype, name)
            np.testing.assert_array_equal(f_py[name], f_rs[name], err_msg=name)

    def test_rs_type(self):
        self._logger.info('TestCRM.test_rs_type()')
        with sw.BACKEND_RUST:
            with self.assertRaises(TypeError):
                sw.crm.extract(np.zeros((32, 32), dtype='uint8'))
            with self.assertRaises(TypeError):
                sw.crm.extract(np.zeros((32, 32, 3), dtype='float32'))
            with self.assertRaises(NotImplementedError):
                sw.crm.extract(np.zeros((32, 32, 3), dtype='uint8'), implementation=sw.CRM_ORIGINAL)
