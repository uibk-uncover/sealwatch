
import importlib.util
import logging
import numpy as np
import os
from PIL import Image
import sealwatch as sw
import sys
import tempfile
import torch
import types
import unittest
import urllib.request

from . import defs


REFERENCE_URL = 'https://raw.githubusercontent.com/albblgb/Deep-Steganalysis/5bef32f9a9c25eb198b05cdec31e2010826ef0fe/models'


def load_reference(tmpdir: str, batch_size: int):
    """Downloads and imports Zhu-Net implementation by albblgb."""
    os.makedirs(os.path.join(tmpdir, 'models', 'kernels'))
    for f in ['ZhuNet.py', 'kernels/SRM3_3.npy', 'kernels/SRM5_5.npy']:
        urllib.request.urlretrieve(f'{REFERENCE_URL}/{f}', os.path.join(tmpdir, 'models', f))
    # reference reads a config module and loads kernels from relative paths
    config = types.ModuleType('config')
    config.stego_img_channel = 1
    config.mode = 'test'
    config.test_batch_size = batch_size
    sys.modules['config'] = config
    cwd = os.getcwd()
    try:
        os.chdir(tmpdir)
        spec = importlib.util.spec_from_file_location('reference_zhunet', os.path.join('models', 'ZhuNet.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
        del sys.modules['config']
    return module


class TestZhuNet(unittest.TestCase):
    """Test suite for zhunet module."""
    _logger = logging.getLogger(__name__)

    def test_structure(self):
        """Checks the shapes against the Fig. 1 in the paper."""
        self._logger.info('TestZhuNet.test_structure()')
        model = sw.zhunet.ZhuNet().eval()
        #
        x = torch.rand((1, 1, 256, 256), dtype=torch.float32)
        x_pre = model.preprocessing(x)
        self.assertEqual(tuple(x_pre.size()), (1, 30, 256, 256))
        #
        x_sep1 = model.sepconv1(x_pre)
        self.assertEqual(tuple(model.sepconv1.depthwise(x_pre).size()), (1, 60, 256, 256))
        self.assertEqual(tuple(x_sep1.size()), (1, 30, 256, 256))
        x_sep2 = model.sepconv2(x_sep1)
        self.assertEqual(tuple(x_sep2.size()), (1, 30, 256, 256))
        #
        x_block1 = model.block1(x_sep2 + x_pre)
        self.assertEqual(tuple(x_block1.size()), (1, 32, 128, 128))
        x_block2 = model.block2(x_block1)
        self.assertEqual(tuple(x_block2.size()), (1, 32, 64, 64))
        x_block3 = model.block3(x_block2)
        self.assertEqual(tuple(x_block3.size()), (1, 64, 32, 32))
        x_block4 = model.block4(x_block3)
        self.assertEqual(tuple(x_block4.size()), (1, 128, 32, 32))
        #
        x_spp = model.spp(x_block4)
        self.assertEqual(tuple(x_spp.size()), (1, 2688))
        x_zhunet = model(x)
        self.assertEqual(tuple(x_zhunet.size()), (1, 2))

    def test_filters(self):
        """Checks the SRM filters of the preprocessing layer."""
        self._logger.info('TestZhuNet.test_filters()')
        filters_3x3, filters_5x5 = sw.zhunet.build_filters()
        self.assertEqual(filters_3x3.shape, (25, 3, 3))
        self.assertEqual(filters_5x5.shape, (5, 5, 5))
        # high-pass filters, except for cropped third-order (8 filters)
        self.assertEqual((filters_3x3.sum(axis=(1, 2)) != 0).sum(), 8)
        np.testing.assert_array_equal(filters_5x5.sum(axis=(1, 2)), 0)
        # unique
        self.assertEqual(len(np.unique(filters_3x3, axis=0)), 25)
        self.assertEqual(len(np.unique(filters_5x5, axis=0)), 5)

    def test_variable_size(self):
        """Thanks to SPP, other sizes are accepted (Sec. III-D)."""
        self._logger.info('TestZhuNet.test_variable_size()')
        model = sw.zhunet.ZhuNet().eval()
        for size in [224, 256, 512]:
            x = torch.rand((2, 1, size, size), dtype=torch.float32)
            self.assertEqual(tuple(model(x).size()), (2, 2))

    def test_compatible_with_reference(self):
        """Compares against implementation by albblgb."""
        self._logger.info('TestZhuNet.test_compatible_with_reference()')
        batch_size = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                reference = load_reference(tmpdir, batch_size)
            except OSError as e:
                self.skipTest(f'reference not available: {e}')
            # same initial SRM kernels, up to ordering
            model_ref = reference.Zhu_Net()
            filters_3x3, filters_5x5 = sw.zhunet.build_filters()
            for ref, ours in [
                (model_ref.layer1.conv1.weight[:, 0], filters_3x3),
                (model_ref.layer1.conv2.weight[:, 0], filters_5x5),
            ]:
                np.testing.assert_array_equal(
                    np.unique(ref.detach().numpy(), axis=0),
                    np.unique(ours, axis=0),
                )

        # same parameters
        model = sw.zhunet.ZhuNet()
        state_dict = sw.zhunet.convert_state_dict(model_ref.state_dict())
        model.load_state_dict(state_dict, strict=True)
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model_ref.parameters()),
        )

        # same outputs
        torch.manual_seed(12345)
        x = torch.rand((batch_size, 1, 256, 256), dtype=torch.float32)
        for train in [True, False]:
            model.train(train)
            model_ref.train(train)
            with torch.no_grad():
                y = model(x)
                y_ref = model_ref(x.clone())
            torch.testing.assert_close(y, y_ref)

    def test_infere_single(self):
        self._logger.info('TestZhuNet.test_infere_single()')
        model = sw.zhunet.ZhuNet().eval()
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{defs.TEST_IMAGES[0]}.png'))
        y0 = sw.zhunet.infere_single(x0, model=model)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(y0, 1)
