
import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import tempfile
import torch
import unittest

from . import defs


class TestXuNet(unittest.TestCase):
    """Test suite for xunet module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    def test_pretrained(self):
        self._logger.info('TestXuNet.test_pretrained()')
        model = sw.xunet.pretrained()
        model = sw.xunet.pretrained('xunet_hill_04.pt')
        model = sw.xunet.pretrained('xunet_hill_01.pt')
        model = sw.xunet.pretrained('xunet_lsbm_04.pt')
        model = sw.xunet.pretrained('xunet_lsbm_01.pt')

    # @parameterized.expand([[fname] for fname in defs.TEST_IMAGES])
    # def test_infere_xunet(self, fname: str):
    #     self._logger.info(f'TestXuNet.test_load_xunet({fname=})')
    #     #
    #     DEVICE = torch.device('cpu')
    #     model = sw.xunet.pretrained('..', device=DEVICE)  # 'XuNet-LSBM_0.4_optimal-250530132417.pt'
    #     #
    #     x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
    #     y0 = sw.xunet.infere_single(x0, model=model, device=DEVICE)
    #     #
    #     x1 = cl.lsb.simulate(x0, alpha=.4, modify=cl.LSB_MATCHING, seed=12345)
    #     y1 = sw.xunet.infere_single(x1, model=model, device=DEVICE)
    #     print(fname, y0, y1)
