
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
        self._logger.info(f'TestCRM.test_extract({fname})')

        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))

        # extract CRM
        start = time.perf_counter()
        features = sw.crm.extract(x0, Tc=3, implementation=sw.CRM_ORIGINAL)
        end = time.perf_counter()
        self._logger.info(f'extracting CRM from {fname}.png took {end - start} s')
        self.assertEqual(len(sw.tools.flatten(features)), 5404)  # from DDE page

        # compare to Matlab reference
        features_ref = scipy.io.loadmat(FEATURE_DIR / f'{fname}.mat', simplify_cells=True)
        features_ref = {k: v for k, v in features_ref.items() if not k.startswith('_')}
        # same submodules
        assert set(features_ref) == set(features)
        # the submodules have the same values
        for k in features:
            # print(k, np.allclose(f[k], f_ref[k]))
            np.testing.assert_array_equal(features[k], features_ref[k])

        # same after merge
        features_ref = OrderedDict([(k, features_ref[k]) for k in features.keys()])
        np.testing.assert_allclose(
            sw.tools.flatten(features),
            sw.tools.flatten(features_ref),
        )

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_extract_crm2(self, fname):
        self._logger.info(f'TestCRM.test_extract_crm2({fname})')

        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))

        # extract CRM
        start = time.perf_counter()
        features = sw.crm.extract(x0, Tc=2, implementation=sw.CRM_ORIGINAL)
        end = time.perf_counter()
        self._logger.info(f'extracting CRM from {fname}.png took {end - start} s')
        self.assertEqual(len(sw.tools.flatten(features)), 2073)
