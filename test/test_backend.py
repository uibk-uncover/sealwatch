
import itertools
import logging
import numpy as np
import os
from parameterized import parameterized
import sealwatch as sw
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings


class TestBackend(unittest.TestCase):
    """Test suite for backend."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    def test_backend(self):
        self._logger.info('TestBackend.test_backend()')
        self.assertIs(sw.get_backend(), sw.BACKEND_RUST)
        try:
            sw.set_backend(sw.BACKEND_PYTHON)
            self.assertIs(sw.get_backend(), sw.BACKEND_PYTHON)
        finally:
            sw.set_backend(sw.BACKEND_RUST)
        self.assertIs(sw.get_backend(), sw.BACKEND_RUST)

    def test_with(self):
        self._logger.info('TestBackend.test_with()')
        self.assertIs(sw.get_backend(), sw.BACKEND_RUST)
        # switch to python
        with sw.BACKEND_PYTHON:
            self.assertIs(sw.get_backend(), sw.BACKEND_PYTHON)
            # switch to rust
            with sw.BACKEND_RUST:
                self.assertIs(sw.get_backend(), sw.BACKEND_RUST)
            # back to python
            self.assertIs(sw.get_backend(), sw.BACKEND_PYTHON)
        # back to default (rust)
        self.assertIs(sw.get_backend(), sw.BACKEND_RUST)

    @parameterized.expand(itertools.product(
        ['spam', 'spam_rounded', 'srm', 'srmq1', 'crm', 'crm_tc8', 'crm_original', 'dctr', 'dctr_2d', 'gfr'],
        [(40, 40), (37, 53), (8, 8), (7, 9), (5, 5), (4, 30), (2, 2)],
        ['uint8', 'float64'],
    ))
    def test_consistent(self, extractor, shape, dtype):
        """Rust backend gives the same features as Python, where Python works, and fails the same way otherwise."""
        self._logger.info(f'TestBackend.test_consistent({extractor=}, {shape=}, {dtype=})')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, 256, shape).astype(dtype)
        x_color = rng.integers(0, 256, shape + (3,)).astype(dtype)
        extract = {
            'spam': lambda: sw.spam.extract(x),
            'spam_rounded': lambda: sw.spam.extract(x, rounded=True),
            'srm': lambda: sw.srm.extract(x),
            'srmq1': lambda: sw.srm.srmq1.extract(x),
            'crm': lambda: sw.crm.extract(x_color),
            'crm_tc8': lambda: sw.crm.extract(x_color, Tc=8),
            'crm_original': lambda: sw.crm.extract(x_color, implementation=sw.CRM_ORIGINAL),
            'dctr': lambda: sw.dctr.extract(x - 128, 4),
            'dctr_2d': lambda: sw.dctr.extract(x - 128, 4, separable=False),
            'gfr': lambda: sw.gfr.extract(x, num_rotations=4),
        }[extractor]

        def run(backend):
            with backend, warnings.catch_warnings():
                warnings.simplefilter('ignore')  # e.g., NaN features of images smaller than the filters
                try:
                    return extract(), None
                except Exception as e:
                    return None, type(e)

        f_py, e_py = run(sw.BACKEND_PYTHON)
        f_rs, e_rs = run(sw.BACKEND_RUST)
        self.assertIs(e_rs, e_py)
        if e_py is None:
            self.assertEqual(list(f_py), list(f_rs))
            for k in f_py:
                self.assertEqual(f_py[k].dtype, f_rs[k].dtype, k)
                np.testing.assert_array_equal(f_py[k], f_rs[k], err_msg=k)

    def test_rust_unavailable(self):
        """Rust backend falls back to Python, if the extension is not available."""
        self._logger.info('TestBackend.test_rust_unavailable()')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, 256, (40, 40)).astype(np.uint8)
        x_color = rng.integers(0, 256, (40, 40, 3)).astype(np.uint8)
        extractors = [
            lambda: sw.spam.extract(x),
            lambda: sw.srm.extract(x),
            lambda: sw.crm.extract(x_color),
            lambda: sw.dctr.extract(x - 128., 4.),
            lambda: sw.gfr.extract(x),
        ]
        for extract in extractors:
            with sw.BACKEND_PYTHON:
                f_py = extract()
            with sw.BACKEND_RUST, mock.patch.object(sw.tools._backend, '_sealwatch', None):
                f_fallback = extract()
            self.assertEqual(list(f_py), list(f_fallback))
            for k in f_py:
                np.testing.assert_array_equal(f_py[k], f_fallback[k], err_msg=k)

    def test_import_without_rust(self):
        """Without the extension, the package imports, warns, and defaults to Python."""
        self._logger.info('TestBackend.test_import_without_rust()')
        code = (
            'import sys, warnings; '
            'sys.modules["sealwatch._sealwatch"] = None; '  # import fails
            'warnings.simplefilter("always"); '
            'import sealwatch as sw; '
            'print(sw.get_backend().name)'
        )
        res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
        self.assertEqual(res.stdout.strip(), 'BACKEND_PYTHON')
        self.assertIn('Rust extension of sealwatch not available', res.stderr)
