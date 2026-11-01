"""Implementation of Zhu-Net.

Introduced in
R. Zhang, F. Zhu, J. Liu, G. Liu
Depth-wise separable convolutions and multi-level pooling
for an efficient spatial CNN-based steganalysis
IEEE TIFS, 2020.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Sequence, Tuple

from .. import _common
from ...tools import hpf


def build_filters() -> Tuple[np.ndarray, np.ndarray]:
    """Builds the 30 SRM high-pass filters of the preprocessing layer.

    "SQUARE 5x5" and "EDGE 5x5" are kept 5x5,
    the remaining 25 filters are reduced to 3x3 (Sec. III-B1).
    The third-order filters do not fit into 3x3,
    so their central part is used (not zero-sum anymore).

    :return: 25 filters of size 3x3 and 5 filters of size 5x5
    :rtype: tuple
    """
    filters_3x3 = (
        hpf.filter_class_1
        + hpf.filter_class_2
        + [hpf.center_crop(f, 3) for f in hpf.filter_class_3]
        + hpf.filter_edge_3x3
        + [hpf.square_3x3]
    )
    filters_5x5 = hpf.filter_edge_5x5 + [hpf.square_5x5]
    return np.array(filters_3x3), np.array(filters_5x5)


class Preprocessing(nn.Module):
    """Preprocessing layer with SRM-initialized, trainable kernels."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, 25, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, 5, kernel_size=5, padding=2)
        # initialize with SRM filters, optimized during training (Alg. 1)
        filters_3x3, filters_5x5 = build_filters()
        with torch.no_grad():
            self.conv3x3.weight.copy_(torch.from_numpy(filters_3x3)[:, None])
            self.conv5x5.weight.copy_(torch.from_numpy(filters_5x5)[:, None])
            self.conv3x3.bias.zero_()
            self.conv5x5.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Returns stacked residuals of both kernel sizes."""
        return torch.cat([self.conv3x3(x), self.conv5x5(x)], dim=1)


class SepConvBlock(nn.Module):
    """Separable convolution block, Fig. 5(b).

    3x3 depthwise convolution (30 groups), 1x1 pointwise convolution, BN.
    """

    def __init__(
        self,
        channels: int = 30,
        *,
        use_abs: bool = False,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, 2*channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(2*channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.use_abs = use_abs
        self.activation = nn.ReLU() if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Returns depthwise -> [abs] -> pointwise -> BN -> [ReLU]."""
        x = self.depthwise(x)
        if self.use_abs:
            x = torch.abs(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.activation(x)


class BasicBlock(nn.Module):
    """Basic block: 3x3 convolution, BN, ReLU, optional 5x5 average pooling."""

    def __init__(self, in_channels: int, out_channels: int, *, pool: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2) if pool else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Returns conv -> BN -> ReLU -> pooling."""
        return self.pool(self.activation(self.bn(self.conv(x))))


class SPP(nn.Module):
    """Spatial pyramid pooling, Algorithm 2.

    For each level n, the feature maps are average-pooled into n x n bins.
    The outputs are flattened and concatenated, coarsest level first.
    """

    def __init__(self, levels: Sequence[int] = (1, 2, 4)) -> None:
        super().__init__()
        self.levels = tuple(levels)

    @property
    def bins(self) -> int:
        """Number of bins per feature map."""
        return sum(n**2 for n in self.levels)

    def forward(self, x: Tensor) -> Tensor:
        """Returns a fixed-length feature vector for any spatial size."""
        return torch.cat([
            F.adaptive_avg_pool2d(x, n).flatten(start_dim=1)
            for n in self.levels
        ], dim=1)


class ZhuNet(nn.Module):
    """A convolutional neural network for steganalysis.
    Intended for binary classification of stego and cover grayscale images.

    Introduced in
    R. Zhang, F. Zhu, J. Liu, G. Liu.
    Depth-wise separable convolutions and multi-level pooling for an efficient spatial CNN-based steganalysis.
    IEEE Transactions on Information Forensics and Security, 2020.

    :param in_channels: number of channels of the input images, defaults to 1 (grayscale)
    :type in_channels: int
    :param num_classes: number of output classes, defaults to 2 (one-hot binary)
    :type num_classes: int
    :param weight_seed: seed of the random initial weights,
        by default the global random number generator
    :type weight_seed: int
    :return: randomly initialized ZhuNet instance, to be trained
    :rtype: ZhuNet

    :Example:

    >>> model = sw.nn.ZhuNet(in_channels=1, num_classes=2)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2, *, weight_seed: int = None) -> None:
        super().__init__()
        with _common.seeded(weight_seed):
            # preprocessing
            self.preprocessing = Preprocessing(in_channels)
            # separable convolutions
            self.sepconv1 = SepConvBlock(30, use_abs=True)
            self.sepconv2 = SepConvBlock(30, activation=False)
            # basic blocks
            self.block1 = BasicBlock(30, 32)
            self.block2 = BasicBlock(32, 32)
            self.block3 = BasicBlock(32, 64)
            self.block4 = BasicBlock(64, 128, pool=False)
            # classification
            self.spp = SPP(levels=(1, 2, 4))
            self.classifier = nn.Sequential(
                nn.Linear(128 * self.spp.bins, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes),
            )
            self._init_weights()

    def _init_weights(self) -> None:
        """He initialization of the convolutions, except the preprocessing."""
        for name, m in self.named_modules():
            if name.startswith('preprocessing'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """Returns the feature maps of the last basic block."""
        x = self.preprocessing(x)
        # sepconv blocks with residual connection
        x = F.relu(self.sepconv2(self.sepconv1(x)) + x)
        # basic blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass, returns logits.

        :param x: images of shape [B, C, H, W] in [0, 1],
            where B is the batch size, C is the number of channels (in_channels),
            and H and W are the height and width of the images
        :type x: torch.Tensor
        :return: logits of shape [B, num_classes]
        :rtype: torch.Tensor

        :Example:

        >>> x = torch.from_numpy(
        ...     x0.astype('float32') / 255.  # normalize to [0, 1]
        ... )[None, None]  # [B, C, H, W]
        >>> logits = model(x)  # [B, 2]
        >>> y = torch.softmax(logits, dim=1)[:, 1]  # stego probability
        """
        x = self.forward_features(x)
        x = self.spp(x)
        return self.classifier(x)


def convert_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Converts state dict of the Zhu-Net implementation by albblgb.

    https://github.com/albblgb/Deep-Steganalysis/blob/main/models/ZhuNet.py

    :param state_dict: state dict of Zhu_Net
    :type state_dict: dict
    :return: state dict of ZhuNet
    :rtype: dict
    """
    prefixes = {
        'layer1.conv1.': 'preprocessing.conv3x3.',
        'layer1.conv2.': 'preprocessing.conv5x5.',
        'layer2.conv1_1.': 'sepconv1.pointwise.',
        'layer2.conv1.': 'sepconv1.depthwise.',
        'layer2.bn1.': 'sepconv1.bn.',
        'layer2.conv2_1.': 'sepconv2.pointwise.',
        'layer2.conv2.': 'sepconv2.depthwise.',
        'layer2.bn2.': 'sepconv2.bn.',
        'layer2.conv_layer.0.': 'block1.',
        'layer2.conv_layer.1.': 'block2.',
        'layer2.conv_layer.2.': 'block3.',
        'layer2.conv_layer.3.': 'block4.',
        'layer2.classfier.': 'classifier.',
    }
    converted = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        for old, new in prefixes.items():
            if k.startswith(old):
                converted[new + k[len(old):]] = v
                break
        else:
            raise KeyError(f'unexpected key {k}')
    return converted


def infere_single(
    x: np.ndarray,
    model: ZhuNet,
    *,
    device: torch.device = None,
) -> np.ndarray:
    """Runs inference for a single image.

    :param x: grayscale image of shape [H, W] in [0, 255]
    :type x: np.ndarray
    :param model: trained model in eval mode
    :type model: ZhuNet
    :param device: torch device to move the model to,
        by default the model stays where it is
    :type device: torch.device
    :return: stego probability
    :rtype: np.ndarray
    """
    return _common.infere_single(x, model, device=device)
