"""Implementation of Xunet.

Based on IEEE Signal Processing Letter 2016 paper:
Guanshuo Xu, Han-Zhou Wu, Yun-Qing Shi
CNN tailored to steganalysis, with facilitated statistical modeling.

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


class ImageProcessing(nn.Module):
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


class ConvBlock(nn.Module):
    """This class returns a building block for XuNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        use_abs: bool = False,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        if activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU(inplace=True)

        self.use_abs = use_abs
        self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns conv -> batch_norm -> activation -> pooling."""
        x = self.conv(inp)
        x = self.batch_norm(x)
        if self.use_abs:
            x = torch.abs(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

    def to(self, *args, **kw):
        super().to(*args, **kw)
        self.conv = self.conv.to(*args, **kw)
        self.batch_norm = self.batch_norm.to(*args, **kw)
        return self


class XuNet(nn.Module):
    """
    XuNet: A convolutional neural network for steganalysis.

    Architecture:
    - Preprocessing with high pass filter
    - 5 convolutional layers with batch normalization and activation
    - Global average pooling
    - Fully connected layers with softmax output

    Intended for binary classification of stego and cover images.
    """

    def __init__(self) -> None:
        super().__init__()
        self.image_processing = ImageProcessing()
        self.layer1 = ConvBlock(1, 8, kernel_size=5, activation="tanh", use_abs=True)
        self.layer2 = ConvBlock(8, 16, kernel_size=5, activation="tanh")
        self.layer3 = ConvBlock(16, 32, kernel_size=1)
        self.layer4 = ConvBlock(32, 64, kernel_size=1)
        self.layer5 = ConvBlock(64, 128, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Forward pass"""
        x = self.image_processing(image)

        # Pass through convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Global average pooling
        x = self.pooling(x)

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def to(self, *args, **kw):
        super().to(*args, **kw)
        self.image_processing = self.image_processing.to(*args, **kw)
        self.layer1 = self.layer1.to(*args, **kw)
        self.layer2 = self.layer2.to(*args, **kw)
        self.layer3 = self.layer3.to(*args, **kw)
        self.layer4 = self.layer4.to(*args, **kw)
        self.layer5 = self.layer5.to(*args, **kw)
        self.fc.to(*args, **kw)
        return self


def pretrained(
    model_path: str = None,
    model_name: str = 'xunet_lsbm_01.pt',
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
        model_url = f'https://github.com/uibk-uncover/sealwatch/releases/download/2025.07/{model_name}'
        cache_dir = Path(torch.hub.get_dir()) / 'sealwatch'
        resume_model_file = tools.networks.download_if_missing(model_url, cache_dir / model_name)
    else:
        resume_model_file = Path(model_path) / model_name

    # load weights
    state_dict = torch.load(resume_model_file, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=strict)
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

    # infere
    logit = model(x_)
    y_ = torch.nn.functional.softmax(logit, dim=1)[0, 1]

    # convert back to numpy
    y = y_.detach().cpu().numpy()
    return y
