
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
        self._logger.info(f'TestSpam.test_extract_rs({fname})')
        #
        path = defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'
        x0 = np.array(Image.open(path))
        #
        f_ref = sw.tools.flatten(sw.spam.extract(x0))
        with sw.BACKEND_RUST:
            f = sw.tools.flatten(sw.spam.extract(x0))
        np.testing.assert_allclose(f, f_ref)

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
