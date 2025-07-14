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
import torchvision.transforms as transforms
from typing import Callable

from .. import tools


class HPF(nn.Module):
    """Computes convolution with KV filter over the input tensor."""

    def __init__(self) -> None:
        """Constructor"""
        super().__init__()
        self.kv_filter = (
            torch.FloatTensor(
                [
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-2.0, 8.0, -12.0, 8.0, -2.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                ]
            ).view(1, 1, 5, 5)
            / 12.0
        )

    def forward(self, inp: Tensor) -> Tensor:
        """Returns tensor convolved with KV filter"""
        return F.conv2d(inp, self.kv_filter, stride=1, padding=2)

    def to(self, *args, **kw):
        super().to(*args, **kw)
        self.kv_filter = self.kv_filter.to(*args, **kw)
        return self


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

    def to(self, *args, **kw):
        super().to(*args, **kw)
        self.convolutional = self.convolutional.to(*args, **kw)
        self.bn = self.bn.to(*args, **kw)
        return self


class XuNet(nn.Module):
    """
    XuNet: A convolutional neural network for steganalysis.

    Architecture:
    - Preprocessing with high pass filter
    - 5 convolutional groups with batch normalization, activation, and pooling
    - Fully connected layers with softmax output

    Intended for binary classification of stego and cover images.
    """

    def __init__(self) -> None:
        super().__init__()
        # HPF layer
        self.hpf = HPF()
        # Convolutional groups
        self.group1 = ConvGroup(1, 8, kernel_size=5, activation='tanh', use_abs=True)
        self.group2 = ConvGroup(8, 16, kernel_size=5, activation='tanh')
        self.group3 = ConvGroup(16, 32, kernel_size=1)
        self.group4 = ConvGroup(32, 64, kernel_size=1)
        self.group5 = ConvGroup(64, 128, kernel_size=1, global_pool=True)
        # Fully connected layer
        self.fc = nn.Linear(128, 2)

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

    def to(self, *args, **kw):
        super().to(*args, **kw)
        self.hpf = self.hpf.to(*args, **kw)
        self.group1 = self.group1.to(*args, **kw)
        self.group2 = self.group2.to(*args, **kw)
        self.group3 = self.group3.to(*args, **kw)
        self.group4 = self.group4.to(*args, **kw)
        self.group5 = self.group5.to(*args, **kw)
        self.fc = self.fc.to(*args, **kw)
        return self


def pretrained(
    model_path: str = None,
    model_name: str = 'XuNet-LSBM_0.4_lsb-250714133836.pt',
    *,
    device: torch.nn.Module = torch.device('cpu'),
    strict: bool = True,
) -> torch.nn.Module:
    """Loads pretrained model. Downloads if missing.

    :param model_path: local path to the model
    :type model_path: str
    :param model_name: filename of the model
    :type model_name: str
    :param device: torch device
    :type device: torch.nn.Module
    :return: loaded XuNet Model
    :rtype: torch.nn.Module
    """
    # model
    model = XuNet().to(dtype=torch.float32, device=device)

    # download if needed
    if model_path is None:
        # raise NotImplementedError('pretrained architectures are not yet available')
        model_url = f'https://github.com/uibk-uncover/sealwatch/releases/download/2025.07/{model_name}'
        cache_dir = Path(torch.hub.get_dir()) / 'sealwatch'
        resume_model_file = tools.networks.download_if_missing(model_url, cache_dir / model_name)
    else:
        resume_model_file = Path(model_path) / model_name

    # load weights
    state_dict = torch.load(resume_model_file, map_location=device, weights_only=False)
    # state_dict = torch.load(resume_model_file, weights_only=True)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    logging.info(f'model {model_name} loaded from {resume_model_file}')
    return model


def infere_single(
    x: np.ndarray,
    model: Callable = None,
    *,
    device: torch.nn.Module = torch.device('cpu'),
) -> np.ndarray:
    """Runs inference for a single image.

    :param x: image
    :type x:
    :param model:
    :type model:
    :param device:
    :type device:
    :return:
    :rtype:
    """
    # prepare data
    x = (x / 255.)[None, None]
    x_ = torch.from_numpy(x).to(dtype=torch.float32, device=device)
    # get model
    model = model.to(dtype=torch.float32, device=device)

    #
    with torch.no_grad():
        # infere
        logit = model(x_)
        y_ = torch.nn.functional.softmax(logit, dim=1)[0, 1]
        # convert back to numpy
        y = y_.detach().cpu().numpy()
    #
    return y
