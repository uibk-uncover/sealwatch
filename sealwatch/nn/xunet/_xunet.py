"""Implementation of Xunet.

Intruduced in
Guanshuo Xu, Han-Zhou Wu, Yun-Qing Shi
Structural design of convolutional neural networks for steganalysis
IEEE Signal Processing Letters, 2016

Inspired by implementation by Brijesh Singh.

Author: Max Ninow
Affiliation: University of Innsbruck
"""

import logging
import numpy as np
from pathlib import Path
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union

from .. import _common
from ...tools import hpf


class HPF(nn.Module):
    """Computes convolution with KV filter over each channel of the input tensor."""

    def __init__(self, in_channels: int = 1) -> None:
        """Constructor"""
        super().__init__()
        # fixed filter, follows .to(), predefined unless in the checkpoint
        kv_filter = torch.from_numpy(hpf.normalized_square_5x5).view(1, 1, 5, 5)
        self.register_buffer('kv_filter', kv_filter.repeat(in_channels, 1, 1, 1))
        self.register_load_state_dict_pre_hook(_common.keep_missing_buffers)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns tensor convolved with KV filter"""
        return F.conv2d(inp, self.kv_filter, stride=1, padding=2, groups=self.kv_filter.size(0))


class ConvGroup(nn.Module):
    """This class returns a building block for XuNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        use_abs: bool = False,
        global_pool: bool = False,
    ) -> None:
        super().__init__()

        self.convolutional = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        self.use_abs = use_abs
        if global_pool:
            self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        else:
            self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns conv -> batch_norm -> activation -> pooling."""
        x = self.convolutional(inp)
        x = self.bn(x)
        if self.use_abs:
            x = torch.abs(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class XuNet(nn.Module):
    """
    XuNet: A convolutional neural network for steganalysis.

    Architecture:
    - Preprocessing with high pass filter
    - 5 convolutional groups with batch normalization, activation, and pooling
    - Fully connected layers with softmax output

    Intended for binary classification of stego and cover images.
    """

    PRETRAINED_URL = 'https://github.com/uibk-uncover/sealwatch/releases/download/2025.07/XuNet-LSBM_0.4_lsb-250714133836.pt'

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        # HPF layer
        self.hpf = HPF(in_channels)
        # Convolutional groups
        self.group1 = ConvGroup(in_channels, 8, kernel_size=5, activation='tanh', use_abs=True)
        self.group2 = ConvGroup(8, 16, kernel_size=5, activation='tanh')
        self.group3 = ConvGroup(16, 32, kernel_size=1)
        self.group4 = ConvGroup(32, 64, kernel_size=1)
        self.group5 = ConvGroup(64, 128, kernel_size=1, global_pool=True)
        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        x = self.hpf(x)

        # Pass through convolutional layers
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @classmethod
    def pretrained(
        cls,
        path: Union[str, Path] = None,
        *,
        device: torch.device = torch.device('cpu'),
        strict: bool = True,
        **kw,
    ) -> 'XuNet':
        """Loads pretrained model. Downloads if missing.

        The default weights are trained against LSB matching at 0.4 bpp.

        :param path: local path to the weights,
            by default downloads them into the torch hub cache
        :type path: str or pathlib.Path
        :param device: torch device
        :type device: torch.device
        :param strict: whether the weights must match exactly
        :type strict: bool
        :return: loaded model in eval mode
        :rtype: XuNet
        """
        model = cls(**kw)
        source = cls.PRETRAINED_URL if path is None else path
        state_dict = _common.load_state_dict(source)
        model.load_state_dict(state_dict, strict=strict)
        logging.info(f'using pretrained weights from {source}')
        return model.to(device).eval()


def pretrained(*args, **kw) -> XuNet:
    return XuNet.pretrained(*args, **kw)


def infere_single(
    x: np.ndarray,
    model: XuNet = None,
    *,
    device: torch.device = None,
) -> np.ndarray:
    """Runs inference for a single image.

    :param x: grayscale image of shape [H, W] in [0, 255]
    :type x: np.ndarray
    :param model: model in eval mode,
        by default loads :func:`XuNet.pretrained` (pass the model for repeated calls)
    :type model: XuNet
    :param device: torch device to move the model to,
        by default the model stays where it is
    :type device: torch.device
    :return: stego probability
    :rtype: np.ndarray
    """
    if model is None:
        model = XuNet.pretrained()
    return _common.infere_single(x, model, device=device)
