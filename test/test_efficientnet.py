
import itertools
import logging
import numpy as np
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import torch
import unittest

from . import defs

try:
    import timm
except ImportError:  # test-time dependency
    timm = None


# our class, timm variant
VARIANTS = [
    ('b0', sw.efficientnet.B0, 'efficientnet_b0'),
    ('b4', sw.efficientnet.B4, 'efficientnet_b4'),
]


@unittest.skipIf(timm is None, 'timm not installed')
class TestEfficientNet(unittest.TestCase):
    """EfficientNet matches the implementation of timm."""
    _logger = logging.getLogger(__name__)

    @staticmethod
    def _models(variant: str, in_channels: int, **kw):
        """Our model and the timm model, with the weights of timm."""
        cls, timm_variant = {v: (c, t) for v, c, t in VARIANTS}[variant]
        # timm has the stem stride
        model = cls(in_channels=in_channels, no_stem_stride=False, **kw)
        model_timm = timm.create_model(timm_variant, in_chans=in_channels, num_classes=2, **kw)
        model.load_state_dict(model_timm.state_dict(), strict=True)
        return model, model_timm

    @staticmethod
    def _inputs(in_channels: int) -> torch.Tensor:
        """Random inputs and test images, as a batch."""
        rng = np.random.default_rng(12345)
        noise = rng.normal(0, 1, (2, 96, 96, in_channels))
        if in_channels == 1:
            images = [np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{f}.png'))[..., None] for f in ['seal1', 'seal2']]
        else:
            images = [np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{f}.png')) for f in ['seal1', 'seal2']]
        images = [x[:96, :96, :in_channels] / 255. for x in images]
        x = np.concatenate([noise, np.stack(images)])
        return torch.from_numpy(x).permute(0, 3, 1, 2).float()

    @parameterized.expand(itertools.product([v for v, _, _ in VARIANTS], [1, 3]))
    def test_parameters(self, variant: str, in_channels: int):
        """Same parameters and buffers, by name, order, shape and type."""
        self._logger.info(f'TestEfficientNet.test_parameters({variant=}, {in_channels=})')
        cls, timm_variant = {v: (c, t) for v, c, t in VARIANTS}[variant]
        state_dict = cls(in_channels=in_channels).state_dict()
        state_dict_timm = timm.create_model(timm_variant, in_chans=in_channels, num_classes=2).state_dict()
        self.assertEqual(list(state_dict), list(state_dict_timm))
        for k in state_dict:
            self.assertEqual(state_dict[k].shape, state_dict_timm[k].shape, k)
            self.assertEqual(state_dict[k].dtype, state_dict_timm[k].dtype, k)
        self.assertEqual(
            sum(p.numel() for p in cls(in_channels=in_channels).parameters()),
            sum(p.numel() for p in timm.create_model(timm_variant, in_chans=in_channels, num_classes=2).parameters()),
        )

    @parameterized.expand(itertools.product([v for v, _, _ in VARIANTS], [1, 3]))
    def test_eval(self, variant: str, in_channels: int):
        """Same outputs in eval mode, for random inputs and test images."""
        self._logger.info(f'TestEfficientNet.test_eval({variant=}, {in_channels=})')
        model, model_timm = self._models(variant, in_channels)
        x = self._inputs(in_channels)
        with torch.no_grad():
            torch.testing.assert_close(model.eval()(x), model_timm.eval()(x), rtol=0, atol=0)

    @parameterized.expand(itertools.product([v for v, _, _ in VARIANTS], [1, 3]))
    def test_train(self, variant: str, in_channels: int):
        """Same outputs, gradients and batch statistics in training mode, with dropout and stochastic depth."""
        self._logger.info(f'TestEfficientNet.test_train({variant=}, {in_channels=})')
        model, model_timm = self._models(variant, in_channels, drop_rate=.3, drop_path_rate=.2)
        x = self._inputs(in_channels)
        torch.manual_seed(12345)
        y = model.train()(x)
        torch.manual_seed(12345)
        y_timm = model_timm.train()(x)
        torch.testing.assert_close(y, y_timm, rtol=0, atol=0)
        # gradients
        y.square().sum().backward()
        y_timm.square().sum().backward()
        parameters = dict(model.named_parameters())
        for name, p_timm in model_timm.named_parameters():
            torch.testing.assert_close(parameters[name].grad, p_timm.grad, rtol=0, atol=0, msg=name)
        # running statistics of batch normalization
        state_dict, state_dict_timm = model.state_dict(), model_timm.state_dict()
        for k in state_dict:
            torch.testing.assert_close(state_dict[k], state_dict_timm[k], rtol=0, atol=0, msg=k)

    def test_no_stem_stride(self):
        """Without the stem stride, as timm with the stride of the stem convolution removed."""
        self._logger.info('TestEfficientNet.test_no_stem_stride()')
        model = sw.efficientnet.B0(in_channels=1)
        model_timm = timm.create_model('efficientnet_b0', in_chans=1, num_classes=2)
        model_timm.conv_stem.stride = (1, 1)
        model.load_state_dict(model_timm.state_dict(), strict=True)
        x = self._inputs(1)
        with torch.no_grad():
            torch.testing.assert_close(model.eval()(x), model_timm.eval()(x), rtol=0, atol=0)

    @parameterized.expand([[1], [2], [3], [4]])
    def test_adapt_input_conv(self, in_channels: int):
        self._logger.info(f'TestEfficientNet.test_adapt_input_conv({in_channels=})')
        weight = torch.randn(32, 3, 3, 3)
        torch.testing.assert_close(
            sw.efficientnet.EfficientNet.adapt_input_conv(in_channels, weight),
            timm.models.adapt_input_conv(in_channels, weight),
            rtol=0, atol=0,
        )

    def test_pretrained(self):
        """ImageNet weights equal the pretrained weights of timm, except for the classifier."""
        self._logger.info('TestEfficientNet.test_pretrained()')
        model = sw.efficientnet.B0.pretrained(in_channels=1)
        model_timm = timm.create_model('efficientnet_b0', pretrained=True, in_chans=1, num_classes=2)
        state_dict, state_dict_timm = model.state_dict(), model_timm.state_dict()
        self.assertEqual(list(state_dict), list(state_dict_timm))
        for k in state_dict:
            if not k.startswith('classifier.'):
                torch.testing.assert_close(state_dict[k], state_dict_timm[k], rtol=0, atol=0, msg=k)


__all__ = ['TestEfficientNet']
