
import logging
import sealwatch as sw
import torch
import unittest


class TestSRNet(unittest.TestCase):
    """Test suite for srnet module."""
    _logger = logging.getLogger(__name__)

    def test_structure(self):
        """Checks the shapes against the Fig. 1 and Sec. II-A in the paper."""
        self._logger.info('TestSRNet.test_structure()')
        model = sw.srnet.SRNet().eval()
        #
        x = torch.rand((1, 1, 256, 256), dtype=torch.float32)
        x_l1 = model.block1[0](x)
        self.assertEqual(tuple(x_l1.size()), (1, 64, 256, 256))
        x_l2 = model.block1[1](x_l1)
        self.assertEqual(tuple(x_l2.size()), (1, 16, 256, 256))
        # unpooled
        x_l7 = model.block2(x_l2)
        self.assertEqual(tuple(x_l7.size()), (1, 16, 256, 256))
        # pooled
        x_l = x_l7
        for layer, size in zip(model.block3, [(16, 128), (64, 64), (128, 32), (256, 16)]):
            x_l = layer(x_l)
            self.assertEqual(tuple(x_l.size()), (1, size[0], size[1], size[1]))
        # global pooling
        x_l12 = model.block4(x_l)
        self.assertEqual(tuple(x_l12.size()), (1, 512, 1, 1))
        #
        y = model(x)
        self.assertEqual(tuple(y.size()), (1, 2))

    def test_shortcuts(self):
        """Type 2 has identity shortcut, type 3 convolutional shortcut, no ReLU after."""
        self._logger.info('TestSRNet.test_shortcuts()')
        model = sw.srnet.SRNet().eval()
        layer = model.block2[0]
        x = torch.randn((1, 16, 32, 32))
        with torch.no_grad():
            y = layer(x)
            main = layer.norm(layer.conv(layer.layer1(x)))
        torch.testing.assert_close(y, main + x)
        self.assertTrue((y < 0).any())
