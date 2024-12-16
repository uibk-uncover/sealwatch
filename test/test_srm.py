"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from collections import OrderedDict
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import re
import scipy.io
import sealwatch as sw
import sys
import tempfile
import time
import unittest

sys.path.append('test')
import defs

FEATURE_DIR = defs.ASSETS_DIR / 'features_matlab' / 'srm'
# FEATURE_DIR = defs.ASSETS_DIR / 'features_matlab' / 'srm'


class TestSRM(unittest.TestCase):
    """Test suite for srm module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    # @parameterized.expand([[f] for f in ['00001', '00002', '00003', '00004', '00005']])
    # @parameterized.expand([[f] for f in ['00001']])  # takes too long
    def test_extract_q1(self, fname):
        #
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        start = time.perf_counter()
        features = sw.srmq1.extract(x0)
        end = time.perf_counter()
        self._logger.info(f'extracting SRMQ1 from {fname}.jpeg took {end - start} s')

        #
        self.assertEqual(len(sw.tools.flatten(features)), 12753)  # from DDE page

        # compare to reference
        features_ref = scipy.io.loadmat(FEATURE_DIR / f'{fname}.mat', simplify_cells=True)
        features_ref = OrderedDict([(k, features_ref[k]) for k in features_ref.keys() if k.endswith('_q1')])

        for submodel, f in features.items():
            f_ref = features_ref[submodel].flatten()

            # Compare to Matlab features
            # print(submodel, np.allclose(f, f_ref))
            np.testing.assert_allclose(f, f_ref)

        features_ref = OrderedDict([(k, features_ref[k]) for k in features.keys()])
        np.testing.assert_allclose(
            sw.tools.flatten(features),
            sw.tools.flatten(features_ref)
        )

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_extract(self, fname):
        self._logger.info(f'TestSRM.test_extract({fname})')
        # print(defs.COVER_DIR)
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        start = time.perf_counter()
        features = sw.srm.extract(x0)
        end = time.perf_counter()
        self._logger.info(f'extracting SRM from {fname}.jpeg took {end - start} s')

        # check length
        self.assertEqual(len(sw.tools.flatten(features)), 34671)  # from DDE page

        # compare to Matlab reference
        features_ref = scipy.io.loadmat(FEATURE_DIR / f'{fname}.mat', simplify_cells=True)
        for submodel, f in features.items():
            f_ref = features_ref[submodel].flatten()

            # Compare to Matlab features
            # print(submodel, np.allclose(f, f_ref))
            np.testing.assert_allclose(f, f_ref)

        # same after merge
        features_ref = OrderedDict([(k, features_ref[k]) for k in features.keys()])
        np.testing.assert_allclose(
            sw.tools.flatten(features),
            sw.tools.flatten(features_ref)
        )


__all__ = ["TestSRM"]
