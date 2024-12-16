""""""

import conseal as cl
import jpeglib
import logging
import numpy as np
import os
from parameterized import parameterized
import sealwatch as sw
import sys
import tempfile
import time
import unittest

sys.path.append('test')
import defs


class TestF5(unittest.TestCase):
    """Test suite for hcfcom module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_attack(self, fname):
        self._logger.info(f'TestF5.test_extract_hcfcom({fname})')

        # load cover
        jpeg0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
        y0, qt = jpeg0.Y, jpeg0.get_component_qt(0)

        # embed
        y1 = cl.F5.simulate_single_channel(y0, alpha=.4, seed=12345)
        beta = (y0 != y1).sum() / cl.tools.nzAC(y0)

        # attack
        beta_hat0 = sw.F5.attack(y0, qt=qt, crop=(3, -5))
        beta_hat1 = sw.F5.attack(y1, qt=qt, crop=(3, -5))

        #
        self.assertLess(np.abs(beta - beta_hat1), .08)
        self.assertLess(beta_hat0, .08)
