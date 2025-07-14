
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

    def test_initialize(self):
        """Checks the shapes against the Fig. 1 in the paper."""
        self._logger.info('TestXuNet.test_initialize()')
        model = sw.xunet.XuNet()
        #
        x = torch.ones((1, 1, 512, 512), dtype=torch.float32)
        x_hpf = model.hpf(x)
        self.assertEqual(tuple(x_hpf.size()), (1, 1, 512, 512))
        #
        x_conv1 = model.group1.convolutional(x_hpf)
        self.assertEqual(tuple(x_conv1.size()), (1, 8, 512, 512))
        x_group1 = model.group1(x_hpf)
        self.assertEqual(tuple(x_group1.size()), (1, 8, 256, 256))
        #
        x_conv2 = model.group2.convolutional(x_group1)
        self.assertEqual(tuple(x_conv2.size()), (1, 16, 256, 256))
        x_group2 = model.group2(x_group1)
        self.assertEqual(tuple(x_group2.size()), (1, 16, 128, 128))
        #
        x_conv3 = model.group3.convolutional(x_group2)
        self.assertEqual(tuple(x_conv3.size()), (1, 32, 128, 128))
        x_group3 = model.group3(x_group2)
        self.assertEqual(tuple(x_group3.size()), (1, 32, 64, 64))
        #
        x_conv4 = model.group4.convolutional(x_group3)
        self.assertEqual(tuple(x_conv4.size()), (1, 64, 64, 64))
        x_group4 = model.group4(x_group3)
        self.assertEqual(tuple(x_group4.size()), (1, 64, 32, 32))
        #
        x_conv5 = model.group5.convolutional(x_group4)
        self.assertEqual(tuple(x_conv5.size()), (1, 128, 32, 32))
        x_group5 = model.group5(x_group4)
        self.assertEqual(tuple(x_group5.size()), (1, 128, 1, 1))
        #
        x_xunet = model(x)
        self.assertEqual(tuple(x_xunet.size()), (1, 2))

    def test_pretrained(self):
        self._logger.info('TestXuNet.test_pretrained()')
        sw.xunet.pretrained()

    @parameterized.expand([[fname] for fname in defs.TEST_IMAGES])
    def test_infere_xunet(self, fname: str):
        self._logger.info(f'TestXuNet.test_load_xunet({fname=})')
        #
        DEVICE = torch.device('cpu')
        model = sw.xunet.pretrained(device=DEVICE)

        #
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        y0 = sw.xunet.infere_single(x0, model=model, device=DEVICE)
        #
        x1 = cl.lsb.simulate(x0, alpha=.4, modify=cl.LSB_MATCHING, seed=12345)
        y1 = sw.xunet.infere_single(x1, model=model, device=DEVICE)
        self.assertLess(y0, y1)
        # print(fname, y0, y1)
