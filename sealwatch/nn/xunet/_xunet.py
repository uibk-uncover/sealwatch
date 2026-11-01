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
    """A convolutional neural network for steganalysis.
    Intended for binary classification of stego and cover grayscale images.

    Introduced in
    Guanshuo Xu, Han-Zhou Wu, Yun-Qing Shi.
    Structural design of convolutional neural networks for steganalysis.
    IEEE Signal Processing Letters, 2016.

    :param in_channels: number of channels of the input images, defaults to 1 (grayscale)
    :type in_channels: int
    :param num_classes: number of output classes, defaults to 2 (one-hot binary)
    :type num_classes: int
    :param weight_seed: seed of the random initial weights,
        by default the global random number generator
    :type weight_seed: int
    :return: randomly initialized XuNet instance, to be trained or loaded with pretrained weights
    :rtype: XuNet

    :Example:

    >>> model = sw.nn.XuNet(in_channels=1, num_classes=2)
    """

    PRETRAINED_URL = 'https://github.com/uibk-uncover/sealwatch/releases/download/2025.07/XuNet-LSBM_0.4_lsb-250714133836.pt'

    def __init__(self, in_channels: int = 1, num_classes: int = 2, *, weight_seed: int = None) -> None:
        super().__init__()
        with _common.seeded(weight_seed):
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
        reset_classifier: bool = False,
        weight_seed: int = None,
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
        :param reset_classifier: whether to initialize the FC layer randomly,
            instead of loading it, e.g., for fine-tuning
        :type reset_classifier: bool
        :param weight_seed: seed of the random initial weights, i.e., of the FC layer,
            by default the global random number generator
        :type weight_seed: int
        :return: loaded model in eval mode
        :rtype: XuNet

        :Example:

        >>> model = sw.nn.XuNet.pretrained()  # trained against LSB matching
        >>> model = sw.nn.XuNet.pretrained("xunet.pt")  # own trained weights
        >>> y = sw.nn.infere_single(x0, model)  # stego probability
        """
        model = cls(weight_seed=weight_seed, **kw)
        source = cls.PRETRAINED_URL if path is None else path
        state_dict = _common.load_state_dict(source)

        # keep FC weights of the random initialization
        if reset_classifier:
            state_dict['fc.weight'] = model.fc.weight.data
            state_dict['fc.bias'] = model.fc.bias.data

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
