
import logging
import numpy as np
import os
from parameterized import parameterized
from pathlib import Path
import pickle
import sealwatch as sw
import subprocess
import sys
import tempfile
import torch
import unittest
from unittest import mock


# constructor, input size
NETWORKS = [
    ('xunet', lambda **kw: sw.xunet.XuNet(**kw), 64),
    ('ucnet', lambda **kw: sw.ucnet.UcNet(**kw), 64),
    ('srnet', lambda **kw: sw.srnet.SRNet(**kw), 64),
    ('zhunet', lambda **kw: sw.zhunet.ZhuNet(**kw), 64),
    ('b0', lambda **kw: sw.efficientnet.B0(**kw), 64),
]

# networks with fixed filters, state dict key
FIXED_FILTERS = [
    ('xunet', 'hpf.kv_filter'),
    ('ucnet', 'preprocessing.filters'),
]

# networks with default pretrained weights, in channels
PRETRAINED = [
    ('xunet', sw.xunet, 1),
    ('ucnet', sw.ucnet, 3),
    ('srnet', sw.srnet, 3),
]


class Malicious:
    """Unpickling it executes code, unless weights_only."""
    def __reduce__(self):
        return (print, ('code executed',))


class TestNetworks(unittest.TestCase):
    """Test suite for the common interface of the networks."""
    _logger = logging.getLogger(__name__)

    @parameterized.expand([
        (name, in_channels, num_classes)
        for name, _, _ in NETWORKS
        for in_channels in [1, 3]
        for num_classes in [2, 5]
    ])
    def test_constructor(self, name: str, in_channels: int, num_classes: int):
        self._logger.info(f'TestNetworks.test_constructor({name=}, {in_channels=}, {num_classes=})')
        create, size = {n: (c, s) for n, c, s in NETWORKS}[name]
        model = create(in_channels=in_channels, num_classes=num_classes).eval()
        x = torch.rand((2, in_channels, size, size))
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.size()), (2, num_classes))

    @parameterized.expand([[name] for name, _, _ in NETWORKS])
    def test_to(self, name: str):
        """Moving the model also moves fixed filters."""
        self._logger.info(f'TestNetworks.test_to({name=})')
        create, size = {n: (c, s) for n, c, s in NETWORKS}[name]
        model = create().to(dtype=torch.float64).eval()
        for t in list(model.parameters()) + list(model.buffers()):
            if t.is_floating_point():
                self.assertEqual(t.dtype, torch.float64)
        in_channels = 3 if name == 'ucnet' else 1
        with torch.no_grad():
            y = model(torch.rand((1, in_channels, size, size), dtype=torch.float64))
        self.assertEqual(y.dtype, torch.float64)

    @parameterized.expand([[name] for name, _, _ in NETWORKS])
    def test_state_dict_nested(self, name: str):
        """Weights round-trip, also when wrapped in another module."""
        self._logger.info(f'TestNetworks.test_state_dict_nested({name=})')
        create, size = {n: (c, s) for n, c, s in NETWORKS}[name]
        in_channels = 3 if name == 'ucnet' else 1
        x = torch.rand((2, in_channels, size, size))
        torch.manual_seed(12345)
        model1 = torch.nn.Sequential(create()).eval()
        torch.manual_seed(54321)
        model2 = torch.nn.Sequential(create()).eval()
        model2.load_state_dict(model1.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(model1(x), model2(x))
        # unwrapped weights load into the wrapped model's inner model
        model2[0].load_state_dict(model1[0].state_dict(), strict=True)

    @parameterized.expand([[name, key] for name, key in FIXED_FILTERS])
    def test_fixed_filters(self, name: str, key: str):
        """Fixed filters are buffers, loaded if in the checkpoint, predefined otherwise."""
        self._logger.info(f'TestNetworks.test_fixed_filters({name=})')
        create = {n: c for n, c, _ in NETWORKS}[name]
        model = create()
        predefined = model.state_dict()[key].clone()
        # not trainable
        self.assertNotIn(key, dict(model.named_parameters()))
        model.requires_grad_(True)
        self.assertFalse(model.get_buffer(key).requires_grad)
        # missing: predefined kept, also when wrapped
        state_dict = model.state_dict()
        del state_dict[key]
        model2 = create()
        model2.load_state_dict(state_dict, strict=True)
        torch.testing.assert_close(model2.state_dict()[key], predefined)
        wrapped = torch.nn.Sequential(create())
        wrapped.load_state_dict({f'0.{k}': v for k, v in state_dict.items()}, strict=True)
        torch.testing.assert_close(wrapped[0].state_dict()[key], predefined)
        # present: loaded, even if different
        state_dict[key] = predefined * 2
        model2.load_state_dict(state_dict, strict=True)
        torch.testing.assert_close(model2.state_dict()[key], predefined * 2)

    def test_ucnet_legacy_filters(self):
        """UcNet stored the filters as frozen convolution before."""
        self._logger.info('TestNetworks.test_ucnet_legacy_filters()')
        model = sw.ucnet.UcNet()
        state_dict = model.state_dict()
        filters = state_dict.pop('preprocessing.filters').clone()  # shares memory with the buffer
        state_dict['preprocessing.0.weight'] = filters * 2
        model.load_state_dict(state_dict, strict=True)
        torch.testing.assert_close(model.preprocessing.filters, filters * 2)

    def test_efficientnet_state_dict_keys(self):
        """EfficientNet uses the keys of timm."""
        self._logger.info('TestNetworks.test_efficientnet_state_dict_keys()')
        model = sw.efficientnet.B0()
        self.assertIn('conv_stem.weight', model.state_dict())
        self.assertIn('0.conv_stem.weight', torch.nn.Sequential(model).state_dict())
        # keys with backbone prefix are accepted too
        state_dict = {f'backbone.{k}': v for k, v in model.state_dict().items()}
        model.load_state_dict(state_dict, strict=True)

    def test_load_state_dict_weights_only(self):
        self._logger.info('TestNetworks.test_load_state_dict_weights_only()')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'malicious.pt'
            torch.save({'weight': torch.zeros(1), 'payload': Malicious()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                sw.tools.networks.load_state_dict(path)
            path = Path(tmpdir) / 'benign.pt'
            torch.save({'weight': torch.zeros(1)}, path)
            self.assertIn('weight', sw.tools.networks.load_state_dict(path))

    def test_download_interrupted(self):
        """Interrupted download does not leave a broken cached file."""
        self._logger.info('TestNetworks.test_download_interrupted()')

        def interrupted(url, filename, reporthook):
            Path(filename).write_bytes(b'partial')
            raise ConnectionError('interrupted')

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / 'model.pt'
            with mock.patch('urllib.request.urlretrieve', side_effect=interrupted):
                with self.assertRaises(ConnectionError):
                    sw.tools.networks.download_if_missing('https://example.com/model.pt', dest)
            self.assertFalse(dest.exists())

    def test_infere_single_input(self):
        self._logger.info('TestNetworks.test_infere_single_input()')
        rng = np.random.default_rng(12345)
        # grayscale
        model = sw.zhunet.ZhuNet().eval()
        y = sw.zhunet.infere_single(rng.integers(0, 256, (64, 64)), model)
        self.assertTrue(0 <= y <= 1)
        # color
        model = sw.efficientnet.B0(in_channels=3).eval()
        y = sw.efficientnet.infere_single(rng.integers(0, 256, (64, 64, 3)), model)
        self.assertTrue(0 <= y <= 1)
        # model stays float64
        model = sw.zhunet.ZhuNet().to(torch.float64).eval()
        sw.zhunet.infere_single(rng.integers(0, 256, (64, 64)), model)
        self.assertEqual(next(model.parameters()).dtype, torch.float64)

    def test_infere_single_train_mode(self):
        """Inference in train mode would use batch statistics of a single image."""
        self._logger.info('TestNetworks.test_infere_single_train_mode()')
        model = sw.zhunet.ZhuNet().train()
        with self.assertRaises(ValueError):
            sw.zhunet.infere_single(np.zeros((64, 64)), model)

    def test_srnet_warnings(self):
        """Importing srnet does not change the global warning filters."""
        self._logger.info('TestNetworks.test_srnet_warnings()')
        # fresh interpreter, dependencies imported first
        code = (
            'import warnings, h5py, jpeglib, numpy, pandas, PIL, scipy, scipy.special, scipy.stats, '
            'torch, torchvision, tqdm; '
            'filters = list(warnings.filters); '
            'import sealwatch; '
            'assert filters == list(warnings.filters), "warning filters changed"'
        )
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_ucnet_unexpected_key(self):
        self._logger.info('TestNetworks.test_ucnet_unexpected_key()')
        with self.assertRaises(KeyError):
            sw.ucnet.convert_state_dict({'module.group9.weight': torch.zeros(1)})

    def test_ucnet_gabor(self):
        """Gabor kernels in numpy match OpenCV."""
        self._logger.info('TestNetworks.test_ucnet_gabor()')
        try:
            import cv2
        except ImportError:
            self.skipTest('opencv not installed')
        for sigma in [.5, 1., 2.]:
            for theta in np.arange(0, np.pi, np.pi / 8):
                for psi in [0, np.pi / 2]:
                    for size in [3, 5, 7]:
                        ref = cv2.getGaborKernel(
                            (size, size), sigma, theta, sigma / .56, .5, psi,
                            ktype=cv2.CV_32F)
                        kernel = sw.ucnet._ucnet.gabor_kernel(
                            size, sigma=sigma, theta=theta,
                            lambd=sigma / .56, gamma=.5, psi=psi)
                        np.testing.assert_allclose(kernel, ref, rtol=0, atol=1e-6)

    @parameterized.expand([[name] for name, _, _ in PRETRAINED])
    def test_pretrained(self, name: str):
        """Pretrained models are in eval mode, on device, and usable for inference."""
        self._logger.info(f'TestNetworks.test_pretrained({name=})')
        module, in_channels = {n: (m, c) for n, m, c in PRETRAINED}[name]
        model = module.pretrained(device=torch.device('cpu'))
        self.assertFalse(model.training)
        shape = (64, 64) if in_channels == 1 else (64, 64, in_channels)
        x = np.random.default_rng(12345).integers(0, 256, shape)
        y = module.infere_single(x, model)
        self.assertTrue(0 <= y <= 1)

    def test_pretrained_efficientnet(self):
        self._logger.info('TestNetworks.test_pretrained_efficientnet()')
        model = sw.efficientnet.B0.pretrained(in_channels=1)
        self.assertFalse(model.training)
        # round-trip through a local file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'b0.pt')
            torch.save(model.state_dict(), path)
            model2 = sw.efficientnet.B0.pretrained(path, in_channels=1)
        x = torch.rand((1, 1, 64, 64))
        with torch.no_grad():
            torch.testing.assert_close(model(x), model2(x))

    def test_pretrained_srnet_grayscale(self):
        """Grayscale SRNet takes the green channel of the color weights."""
        self._logger.info('TestNetworks.test_pretrained_srnet_grayscale()')
        model1 = sw.srnet.SRNet.pretrained(in_channels=1)
        model3 = sw.srnet.SRNet.pretrained(in_channels=3)
        torch.testing.assert_close(
            model1.block1[0].conv.weight,
            model3.block1[0].conv.weight[:, 1:2],
        )
        torch.testing.assert_close(model1.fc.weight, model3.fc.weight)
