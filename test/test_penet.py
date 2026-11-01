import importlib.util
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import sys
import tempfile
import torch
import types
import unittest
import urllib.request

from . import defs


REFERENCE_URL = 'https://raw.githubusercontent.com/revere7/PENet_Steganalysis/0125ed542abfa13cc83f1a9f0ef1d243e0cf7233'


def load_reference(tmpdir: str):
    """Downloads and imports the PENet implementation by the authors, for arbitrary-size images."""
    for name in ['PENet_Arbitrary.py', 'srm_filter_kernel.py']:
        urllib.request.urlretrieve(f'{REFERENCE_URL}/{name}', os.path.join(tmpdir, name))
    # run on CPU
    path = os.path.join(tmpdir, 'PENet_Arbitrary.py')
    with open(path) as f:
        source = f.read().replace('.cuda()', '')
    with open(path, 'w') as f:
        f.write(source)
    # imported, but not used by the network
    stubs = {
        'einops': types.SimpleNamespace(rearrange=None, repeat=None),
        'einops.layers': types.SimpleNamespace(),
        'einops.layers.torch': types.SimpleNamespace(Rearrange=None),
        'distutils': types.SimpleNamespace(),  # removed in Python 3.12
        'distutils.errors': types.SimpleNamespace(DistutilsModuleError=None),
    }
    # only the stubs are removed afterwards, not the modules imported by the reference
    stubs = {k: v for k, v in stubs.items() if k not in sys.modules}
    sys.modules.update(stubs)
    sys.path.insert(0, tmpdir)
    try:
        spec = importlib.util.spec_from_file_location('reference_penet', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(tmpdir)
        for k in list(stubs) + ['srm_filter_kernel']:
            sys.modules.pop(k, None)
    return module


def images(size: int = 512) -> torch.Tensor:
    """Batch of two test images and a random image, in [0, 255]."""
    x = [np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{f}.png'))[:size, :size] for f in ['seal1', 'seal2']]
    x.append(np.random.default_rng(12345).integers(0, 256, (size, size, 3)))
    return torch.from_numpy(np.stack(x)).permute(0, 3, 1, 2).float()


class TestPENet(unittest.TestCase):
    """Test suite for penet module."""
    _logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        try:
            import cv2  # noqa: F401
            cls.reference = load_reference(cls._tmpdir.name)
        except (ImportError, OSError) as e:
            cls.reference = None
            cls._reference_error = str(e)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _require_reference(self):
        if self.reference is None:
            self.skipTest(f'reference not available: {self._reference_error}')

    def _models(self):
        """Reference with its initial weights, and PENET_ORIGINAL with the same weights."""
        torch.manual_seed(12345)
        reference = self.reference.Net()
        reference.apply(self.reference.initWeights)
        # the position bias is not initialized in the reference
        torch.nn.init.trunc_normal_(reference.entrans.B, std=.02)
        model = sw.nn.PENet(implementation=sw.PENET_ORIGINAL)
        model.load_state_dict(sw.nn.penet.convert_state_dict(reference.state_dict()), strict=True)
        return model, reference

    def test_filters(self):
        """The fixed filters of the stem are the SRM and Gabor filters of the reference."""
        self._logger.info('TestPENet.test_filters()')
        self._require_reference()
        model = sw.nn.PENet()
        torch.testing.assert_close(model.stem.filters, self.reference.HPF().hpf.weight.data, rtol=0, atol=0)

    def test_parameters(self):
        """Same parameters, by shape, as the reference."""
        self._logger.info('TestPENet.test_parameters()')
        self._require_reference()
        reference = self.reference.Net()
        state_dict = sw.nn.penet.convert_state_dict(reference.state_dict())
        model = sw.nn.PENet(implementation=sw.PENET_ORIGINAL)
        self.assertEqual(set(state_dict), set(model.state_dict()))
        for k, v in model.state_dict().items():
            self.assertEqual(v.shape, state_dict[k].shape, k)
        # trainable parameters, without the fixed filters
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in reference.parameters() if p.requires_grad),
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8_292_610)

    def test_eval(self):
        """PENET_ORIGINAL gives the same outputs as the reference, in eval mode."""
        self._logger.info('TestPENet.test_eval()')
        self._require_reference()
        model, reference = self._models()
        x = images()
        with torch.no_grad():
            torch.testing.assert_close(model.eval()(x / 255.), reference.eval()(x))

    def test_train(self):
        """PENET_ORIGINAL gives the same outputs and gradients as the reference, in training mode."""
        self._logger.info('TestPENet.test_train()')
        self._require_reference()
        model, reference = self._models()
        x = images()[:2]
        y, y_ref = model.train()(x / 255.), reference.train()(x)
        torch.testing.assert_close(y, y_ref)
        y.square().sum().backward()
        y_ref.square().sum().backward()
        grads_ref = sw.nn.penet.convert_state_dict({
            name: p.grad for name, p in reference.named_parameters() if p.requires_grad
        })
        grads = {name: p.grad for name, p in model.named_parameters()}
        self.assertEqual(set(grads), set(grads_ref))
        for name in grads:
            # relative to the scale of each gradient, which differ by orders of magnitude;
            # the gradient of the key bias is zero, as the softmax is invariant to it, up to rounding
            scale = grads_ref[name].abs().max().item()
            torch.testing.assert_close(grads[name], grads_ref[name], rtol=1e-4, atol=1e-4 * scale + 1e-10, msg=name)

    @staticmethod
    def _transformer(implementation):
        """Transformer of PENet with sharp attention, as after training, the initial projections are close to zero."""
        transformer = sw.nn.PENet(input_size=256, implementation=implementation, weight_seed=12345).transformer.eval()
        with torch.no_grad():
            for fc in [transformer.fc_q, transformer.fc_k]:
                torch.nn.init.normal_(fc.weight, std=1.)
        return transformer

    def test_batch_independent(self):
        """The output of an image does not depend on the other images in the batch."""
        self._logger.info('TestPENet.test_batch_independent()')
        transformer = self._transformer(sw.PENET_FIX)
        x = torch.randn(3, 256, 16, 16)
        with torch.no_grad():
            torch.testing.assert_close(transformer(x)[:1], transformer(x[:1]))

    def test_batch_dependent_original(self):
        """PENET_ORIGINAL normalizes the queries over the batch, so that the output depends on it."""
        self._logger.info('TestPENet.test_batch_dependent_original()')
        transformer = self._transformer(sw.PENET_ORIGINAL)
        x = torch.randn(3, 256, 16, 16)
        with torch.no_grad():
            self.assertGreater((transformer(x)[:1] - transformer(x[:1])).abs().max(), 1e-3)

    @parameterized.expand([
        (512, (1, 2, 32 * 32, 16 * 16)),
        (256, (1, 2, 16 * 16, 8 * 8)),
        ((320, 640), (1, 2, 20 * 40, 10 * 20)),
    ])
    def test_input_size(self, size, bias_shape):
        """The position bias is sized for the input size."""
        self._logger.info(f'TestPENet.test_input_size({size=})')
        model = sw.nn.PENet(input_size=size).eval()
        self.assertEqual(model.transformer.position_bias.shape, bias_shape)
        x = torch.rand((1, 3) + model.input_size)
        with torch.no_grad():
            self.assertEqual(model(x).shape, (1, 2))

    def test_input_size_mismatch(self):
        self._logger.info('TestPENet.test_input_size_mismatch()')
        with self.assertRaises(ValueError):
            sw.nn.PENet(input_size=256)(torch.rand(1, 3, 512, 512))

    def test_head(self):
        """The linear head has one fully-connected layer instead of two."""
        self._logger.info('TestPENet.test_head()')
        paper = sw.nn.PENet(input_size=64)
        linear = sw.nn.PENet(input_size=64, head='linear')
        self.assertEqual(sum(p.numel() for p in paper.parameters()) - sum(p.numel() for p in linear.parameters()), 5_497_344)
        with self.assertRaises(ValueError):
            sw.nn.PENet(head='gap')

    def test_pretrained_reference(self):
        """Checkpoint of the authors' training loads, with the same outputs."""
        self._logger.info('TestPENet.test_pretrained_reference()')
        self._require_reference()
        _, reference = self._models()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'penet.pt')
            torch.save({'original_state': reference.state_dict(), 'epoch': 1}, path)
            model = sw.nn.PENet.pretrained(path, implementation=sw.PENET_ORIGINAL)
        x = images()[:2]
        with torch.no_grad():
            torch.testing.assert_close(model(x / 255.), reference.eval()(x))

    def test_pretrained(self):
        """Checkpoint of PENet loads unchanged."""
        self._logger.info('TestPENet.test_pretrained()')
        model = sw.nn.PENet(input_size=64, weight_seed=12345).eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'penet.pt')
            torch.save(model.state_dict(), path)
            model_loaded = sw.nn.PENet.pretrained(path, input_size=64)
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(model_loaded(x), model(x), rtol=0, atol=0)

    def test_pretrained_linear(self):
        """Checkpoint with the paper head loads with the linear head, merged."""
        self._logger.info('TestPENet.test_pretrained_linear()')
        model = sw.nn.PENet(input_size=64, weight_seed=12345).eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'penet.pt')
            torch.save(model.state_dict(), path)
            model_linear = sw.nn.PENet.pretrained(path, input_size=64, head='linear')
        self.assertEqual(model_linear.head, 'linear')
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            # float32 rounding of the merged head, see test_merge_head
            torch.testing.assert_close(model_linear(x), model(x), rtol=1e-5, atol=1e-6)

    def test_pretrained_errors(self):
        self._logger.info('TestPENet.test_pretrained_errors()')
        # no published weights
        with self.assertRaises(ValueError):
            sw.nn.PENet.pretrained()
        # linear head cannot be split into two layers
        model_linear = sw.nn.PENet(input_size=64, head='linear')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'penet.pt')
            torch.save(model_linear.state_dict(), path)
            with self.assertRaises(ValueError):
                sw.nn.PENet.pretrained(path, input_size=64)

    def test_merge_head(self):
        """Model with the paper head, merged into the linear head, gives the same outputs."""
        self._logger.info('TestPENet.test_merge_head()')
        model = sw.nn.PENet(input_size=64, weight_seed=12345).double().eval()
        model_linear = sw.nn.PENet(input_size=64, head='linear').double().eval()
        model_linear.load_state_dict(sw.nn.penet.merge_head(model.state_dict()), strict=True)
        x = torch.rand(4, 3, 64, 64, dtype=torch.float64)
        with torch.no_grad():
            # same up to rounding, as (W2 W1) x and W2 (W1 x) are computed in different order
            torch.testing.assert_close(model_linear(x), model(x), rtol=1e-12, atol=1e-12)

    def test_infere_single(self):
        """Single image in [0, 255], the model in [0, 1]."""
        self._logger.info('TestPENet.test_infere_single()')
        model = sw.nn.PENet(input_size=256, weight_seed=12345).eval()
        x = images(256)[0]
        y = sw.nn.penet.infere_single(x.permute(1, 2, 0).numpy(), model)
        with torch.no_grad():
            y_ref = torch.softmax(model(x[None] / 255.), dim=1)[0, 1]
        self.assertAlmostEqual(float(y), float(y_ref), places=5)

    def test_weight_seed(self):
        self._logger.info('TestPENet.test_weight_seed()')
        m1 = sw.nn.PENet(input_size=64, weight_seed=12345)
        m2 = sw.nn.PENet(input_size=64, weight_seed=12345)
        for (k, v1), v2 in zip(m1.state_dict().items(), m2.state_dict().values()):
            torch.testing.assert_close(v1, v2, rtol=0, atol=0, msg=k)

    def test_convert_unexpected_key(self):
        self._logger.info('TestPENet.test_convert_unexpected_key()')
        with self.assertRaises(KeyError):
            sw.nn.penet.convert_state_dict({'group9.weight': torch.zeros(1)})


__all__ = ['TestPENet']
