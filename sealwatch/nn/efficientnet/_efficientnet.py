"""
Implementation of EfficientNet, as described in

M. Tan and Q. V. Le, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
International Conference on Machine Learning (ICML), 2019.
https://arxiv.org/abs/1905.11946

The modules are named as in timm (https://github.com/huggingface/pytorch-image-models),
so that its weights, and checkpoints of the former timm-based implementation, load unchanged.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import logging
import math
from pathlib import Path
import torch
from typing import List, Tuple, Union

from .. import _common


# EfficientNet-B0 baseline network, Table 1 of the paper, without the stem and the head:
# (expansion ratio of MBConv, kernel size, stride, output channels, number of layers)
# The stride of each stage is given by the resolutions in Table 1.
B0_STAGES: List[Tuple[int, int, int, int, int]] = [
    (1, 3, 1, 16, 1),
    (6, 3, 2, 24, 2),
    (6, 5, 2, 40, 2),
    (6, 3, 2, 80, 3),
    (6, 5, 1, 112, 3),
    (6, 5, 2, 192, 4),
    (6, 3, 1, 320, 1),
]
B0_STEM_CHANNELS = 32
B0_HEAD_CHANNELS = 1280

# squeeze-and-excitation, reduction relative to the block input channels
SE_RATIO = .25


def round_channels(channels: float, divisor: int = 8) -> int:
    """Number of channels after width scaling, rounded to a multiple of divisor,
    by at most 10% down, as in the official implementation."""
    rounded = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if rounded < .9 * channels:
        rounded += divisor
    return rounded


def round_layers(layers: int, depth_multiplier: float) -> int:
    """Number of layers after depth scaling, rounded up."""
    return int(math.ceil(layers * depth_multiplier))


class BatchNormAct(torch.nn.BatchNorm2d):
    """Batch normalization, followed by Swish (SiLU) activation, if act is True.

    The parameters are named as of batch normalization, e.g., bn1.weight.
    """
    def __init__(self, channels: int, act: bool = True):
        # PyTorch defaults (eps=1e-5, momentum=.1), as in timm, unlike the official TensorFlow (eps=1e-3)
        super().__init__(channels)
        self.act = torch.nn.SiLU() if act else torch.nn.Identity()

    def forward(self, x):
        return self.act(super().forward(x))


class SqueezeExcite(torch.nn.Module):
    """Squeeze-and-excitation (Hu et al., 2018), with Swish activation and sigmoid gate."""
    def __init__(self, channels: int, reduced_channels: int):
        super().__init__()
        self.conv_reduce = torch.nn.Conv2d(channels, reduced_channels, 1)
        self.act1 = torch.nn.SiLU()
        self.conv_expand = torch.nn.Conv2d(reduced_channels, channels, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = self.conv_expand(self.act1(self.conv_reduce(s)))
        return x * torch.sigmoid(s)


class DropPath(torch.nn.Module):
    """Stochastic depth (Huang et al., 2016), drops the residual branch per sample, when training."""
    def __init__(self, p: float = 0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        # mask scaled by the keep probability, so that the expectation does not change
        keep = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(1 - self.p)
        return x * keep.div_(1 - self.p)


class MBConv(torch.nn.Module):
    """Mobile inverted bottleneck (Sandler et al., 2018) with squeeze-and-excitation.

    MBConv1 (expansion ratio 1) has no expansion convolution, and its modules are named
    as DepthwiseSeparableConv of timm, MBConv6 as InvertedResidual.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expansion: int,
        kernel_size: int,
        stride: int,
        drop_path_rate: float = 0.,
    ):
        super().__init__()
        self.expansion = expansion
        mid_channels = in_channels * expansion
        reduced_channels = round(in_channels * SE_RATIO)
        if expansion == 1:
            # depthwise, SE, pointwise
            self.conv_dw = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size, stride,
                padding=kernel_size // 2, groups=in_channels, bias=False)
            self.bn1 = BatchNormAct(in_channels)
            self.se = SqueezeExcite(in_channels, reduced_channels)
            self.conv_pw = torch.nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn2 = BatchNormAct(out_channels, act=False)
        else:
            # expansion, depthwise, SE, linear projection
            self.conv_pw = torch.nn.Conv2d(in_channels, mid_channels, 1, bias=False)
            self.bn1 = BatchNormAct(mid_channels)
            self.conv_dw = torch.nn.Conv2d(
                mid_channels, mid_channels, kernel_size, stride,
                padding=kernel_size // 2, groups=mid_channels, bias=False)
            self.bn2 = BatchNormAct(mid_channels)
            self.se = SqueezeExcite(mid_channels, reduced_channels)
            self.conv_pwl = torch.nn.Conv2d(mid_channels, out_channels, 1, bias=False)
            self.bn3 = BatchNormAct(out_channels, act=False)
        # residual connection, if the shape does not change
        self.has_skip = stride == 1 and in_channels == out_channels
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        if self.expansion == 1:
            y = self.bn1(self.conv_dw(x))
            y = self.se(y)
            y = self.bn2(self.conv_pw(y))
        else:
            y = self.bn1(self.conv_pw(x))
            y = self.bn2(self.conv_dw(y))
            y = self.se(y)
            y = self.bn3(self.conv_pwl(y))
        if self.has_skip:
            y = self.drop_path(y) + x
        return y


class EfficientNet(torch.nn.Module):
    """EfficientNet, adapted for steganalysis.

    The B0 baseline network is scaled in width and depth by the compound scaling,
    with the coefficients of the subclasses.
    """
    width_multiplier: float = None
    depth_multiplier: float = None
    weights_url: str = None
    """ImageNet weights of timm."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        no_stem_stride: bool = True,
        weight_seed: int = None,
        *,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.,
    ):
        """Constructor.

        :param in_channels: number of input channels
        :type in_channels: int
        :param num_classes: number of output classes
        :type num_classes: int
        :param no_stem_stride: if True, the stem convolution has no stride,
            which keeps the high-frequency stego signal in steganalysis
        :type no_stem_stride: bool
        :param weight_seed: seed of the random initial weights,
            by default the global random number generator
        :type weight_seed: int
        :param drop_rate: dropout rate before the classifier
        :type drop_rate: float
        :param drop_path_rate: stochastic depth rate of the last block,
            the earlier blocks increase linearly from zero
        :type drop_path_rate: float
        """
        super().__init__()
        # stem
        stem_channels = round_channels(B0_STEM_CHANNELS * self.width_multiplier)
        self.conv_stem = torch.nn.Conv2d(
            in_channels, stem_channels, 3, stride=1 if no_stem_stride else 2, padding=1, bias=False)
        self.bn1 = BatchNormAct(stem_channels)

        # MBConv stages
        stages = [
            (expansion, kernel_size, stride, round_channels(channels * self.width_multiplier), round_layers(layers, self.depth_multiplier))
            for expansion, kernel_size, stride, channels, layers in B0_STAGES
        ]
        num_blocks = sum(layers for *_, layers in stages)
        self.blocks = torch.nn.Sequential()
        in_channels, block_idx = stem_channels, 0
        for expansion, kernel_size, stride, out_channels, layers in stages:
            stage = torch.nn.Sequential()
            for layer in range(layers):
                stage.append(MBConv(
                    in_channels,
                    out_channels,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    stride=stride if layer == 0 else 1,  # downsampling by the first layer
                    drop_path_rate=drop_path_rate * block_idx / num_blocks,
                ))
                in_channels, block_idx = out_channels, block_idx + 1
            self.blocks.append(stage)

        # head
        head_channels = round_channels(B0_HEAD_CHANNELS * self.width_multiplier)
        self.conv_head = torch.nn.Conv2d(in_channels, head_channels, 1, bias=False)
        self.bn2 = BatchNormAct(head_channels)
        self.dropout = torch.nn.Dropout(drop_rate)
        self.classifier = torch.nn.Linear(head_channels, num_classes)

        # initial weights
        with torch.random.fork_rng():
            if weight_seed is not None:
                torch.manual_seed(weight_seed)
            self.apply(self._init_weights)

        # checkpoints with the backbone prefix of the former timm-based implementation
        self.register_load_state_dict_pre_hook(self._strip_backbone_prefix)

    @staticmethod
    def _init_weights(module: torch.nn.Module):
        """Initial weights of the official implementation."""
        if isinstance(module, torch.nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels // module.groups
            torch.nn.init.normal_(module.weight, 0, math.sqrt(2. / fan_out))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Linear):
            init_range = 1. / math.sqrt(module.out_features)
            torch.nn.init.uniform_(module.weight, -init_range, init_range)
            torch.nn.init.zeros_(module.bias)

    @staticmethod
    def _strip_backbone_prefix(module, state_dict, prefix, *args):
        for k in [k for k in state_dict if k.startswith(f'{prefix}backbone.')]:
            state_dict[prefix + k[len(f'{prefix}backbone.'):]] = state_dict.pop(k)

    def forward(self, x):
        x = self.bn1(self.conv_stem(x))
        x = self.blocks(x)
        x = self.bn2(self.conv_head(x))
        # global average pooling
        x = x.mean((2, 3))
        return self.classifier(self.dropout(x))

    @staticmethod
    def adapt_input_conv(in_channels: int, weight: torch.Tensor) -> torch.Tensor:
        """Adapts the weights of the RGB stem convolution to in_channels, as in timm.

        Grayscale sums the color channels, other numbers of channels repeat them, scaled.
        """
        if in_channels == 1:
            return weight.sum(dim=1, keepdim=True)
        repeat = int(math.ceil(in_channels / 3))
        return weight.repeat(1, repeat, 1, 1)[:, :in_channels] * (3 / in_channels)

    @classmethod
    def pretrained(
        cls,
        path: Union[str, Path] = None,
        *,
        in_channels: int = 1,
        num_classes: int = 2,
        device: torch.device = torch.device('cpu'),
        strict: bool = True,
        **kw,
    ) -> 'EfficientNet':
        """Loads pretrained model. Downloads if missing.

        By default, loads the ImageNet weights of timm,
        with the input convolution adapted to in_channels,
        and the classifier initialized randomly, i.e., to be fine-tuned.

        :param path: local path to the weights of a trained detector,
            by default downloads the ImageNet weights into the torch hub cache
        :type path: str or pathlib.Path
        :param in_channels: number of input channels
        :type in_channels: int
        :param num_classes: number of output classes
        :type num_classes: int
        :param device: torch device
        :type device: torch.device
        :param strict: whether the weights must match exactly
        :type strict: bool
        :return: loaded model in eval mode
        :rtype: EfficientNet
        """
        # create model
        model = cls(in_channels=in_channels, num_classes=num_classes, **kw)

        if path is None:
            # download weights
            source = cls.weights_url
            state_dict = _common.load_state_dict(source)

            # adapt input channels
            if state_dict['conv_stem.weight'].size(1) != in_channels:
                state_dict['conv_stem.weight'] = cls.adapt_input_conv(in_channels, state_dict['conv_stem.weight'])

            # remove FC
            model_state_dict = model.state_dict()
            state_dict['classifier.weight'] = model_state_dict['classifier.weight']
            state_dict['classifier.bias'] = model_state_dict['classifier.bias']
        else:
            source = path
            state_dict = _common.load_state_dict(source)

        # load weights
        model.load_state_dict(state_dict, strict=strict)
        logging.info(f'using pretrained weights from {source}')
        return model.to(device).eval()


class B0(EfficientNet):
    """EfficientNet-B0, the baseline network."""
    width_multiplier = 1.
    depth_multiplier = 1.
    weights_url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth'


class B4(EfficientNet):
    """EfficientNet-B4.

    The coefficients of the compound scaling, width 1.4 and depth 1.8,
    are of the official implementation, as the paper gives only the scaling rule.
    """
    width_multiplier = 1.4
    depth_multiplier = 1.8
    weights_url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_320-7eb33cd5.pth'
