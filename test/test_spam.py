
import logging
import numpy as np
import os
from parameterized import parameterized
import scipy.io
import sealwatch as sw
import tempfile
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


__all__ = ["TestSpam"]
