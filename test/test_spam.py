
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import scipy.io
import sealwatch as sw
import tempfile
import time
import unittest

from . import defs


FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'spam'


class TestSpam(unittest.TestCase):
    """Test suite for spam module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_extract(self, fname):
        self._logger.info(f'TestSpam.test_extract({fname})')
        #
        path = defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg'
        f = sw.spam.extract_from_file(path)
        f = sw.tools.flatten(f)
        #
        path = FEATURES_DIR / f'{fname}.mat'
        f_ref = scipy.io.loadmat(path)["features"].flatten()
        np.testing.assert_allclose(f, f_ref)


    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_extract_rs(self, fname):
        """Rust implementation is bit-identical to the Python implementation."""
        self._logger.info(f'TestSpam.test_extract_rs({fname})')
        #
        path = defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'
        x0 = np.array(Image.open(path))
        #
        with sw.BACKEND_PYTHON:
            f_ref = sw.tools.flatten(sw.spam.extract(x0))
        with sw.BACKEND_RUST:
            f = sw.tools.flatten(sw.spam.extract(x0))
        np.testing.assert_array_equal(f, f_ref)

    @parameterized.expand([
        (shape, T)
        for shape in ['512x256', '100x400', 'odd']
        for T in [2, 3, 4]
    ])
    def test_extract_rs_nonsquare(self, shape, T):
        """Rust implementation is bit-identical to the Python implementation, also for non-square images,
        where the horizontal and vertical directions have different numbers of triplets."""
        self._logger.info(f'TestSpam.test_extract_rs_nonsquare({shape=}, {T=})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / 'seal1.png'))
        x = {
            '512x256': x0[:, :256],
            '100x400': x0[:100, :400],
            'odd': np.random.default_rng(12345).integers(0, 256, (37, 53)).astype(np.uint8),
        }[shape]
        with sw.BACKEND_PYTHON:
            f_py = sw.spam.extract(x, T=T)
        with sw.BACKEND_RUST:
            f_rs = sw.spam.extract(x, T=T)
        self.assertEqual(list(f_py), list(f_rs))
        for k in f_py:
            np.testing.assert_array_equal(f_py[k], f_rs[k], err_msg=k)

    def test_speedup(self):
        self._logger.info('TestSpam.test_speedup()')
        #
        with sw.BACKEND_RUST:
            start = time.perf_counter()
            for fname in defs.TEST_IMAGES:
                path = defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'
                x0 = np.array(Image.open(path))
                f = sw.tools.flatten(sw.spam.extract(x0))
            end = time.perf_counter()
            print('rust:', end - start, 's')
        #
        with sw.BACKEND_PYTHON:
            start = time.perf_counter()
            for fname in defs.TEST_IMAGES:
                path = defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'
                x0 = np.array(Image.open(path))
                f = sw.tools.flatten(sw.spam.extract(x0))
            end = time.perf_counter()
            print('python:', end - start, 's')

    def test_type(self):
        self._logger.info('TestSpam.test_extract_rs()')
        #
        path = defs.COVER_UNCOMPRESSED_GRAY_DIR / 'seal1.png'
        with sw.BACKEND_RUST:
            x0 = np.array(Image.open(path)).astype('float32')
            try:
                sw.tools.flatten(sw.spam.extract(x0))
            except TypeError:
                raised = True
            else:
                raised = False
        #
        self.assertTrue(raised)



__all__ = ["TestSpam"]
