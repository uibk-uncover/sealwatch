"""Implementation of UcNet architecture, introduced in

K. Wei, W. Luo, S. Tan, J. Huang.
Universal deep network for steganalysis of color image based on channel representation.
IEEE TIFS, 2022.

Inspired by the authors' implementation
https://github.com/revere7/UCNet_Steganalysis/

"""

import numpy as np
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from .. import _common
from ...tools import hpf


def gabor_kernel(
    size: int,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float,
    psi: float,
) -> np.ndarray:
    """Builds a Gabor kernel, equivalent to cv2.getGaborKernel.

    :param size: kernel size, odd
    :type size: int
    :param sigma: standard deviation of the Gaussian envelope
    :type sigma: float
    :param theta: orientation
    :type theta: float
    :param lambd: wavelength of the sinusoid
    :type lambd: float
    :param gamma: spatial aspect ratio
    :type gamma: float
    :param psi: phase offset
    :type psi: float
    :return: kernel of shape [size, size]
    :rtype: np.ndarray
    """
    r = size // 2
    # OpenCV stores kernel(r - y, r - x), i.e., coordinates descend along the axes
    y, x = np.mgrid[r:-r-1:-1, r:-r-1:-1]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-.5 * (x_theta**2 / sigma**2 + y_theta**2 * gamma**2 / sigma**2))
    carrier = np.cos(2 * np.pi / lambd * x_theta + psi)
    return (envelope * carrier).astype(np.float32)


def build_filters():
    """Builds the rich model/Gabor filters.

    30 SRM filters zero-padded to 5x5, and 32 Gabor filters.
    Same filters in the same order as the authors' implementation,
    https://github.com/revere7/UCNet_Steganalysis/
    """
    filters = []
    sigma = [0.5, 1.0]
    phi = [0, np.pi / 2]
    for hpf_item in hpf.all_normalized_hpf_list:
        filters.append(hpf.center_pad(hpf_item, 5))
    for theta in np.arange(0, np.pi, np.pi / 8):  # gabor 0 22.5 45 67.5 90 112.5 135 157.5
        for k in range(2):
            for j in range(2):
                kern = gabor_kernel(
                    5,
                    sigma=sigma[k],
                    theta=theta,
                    lambd=sigma[k] / .56,
                    gamma=.5,
                    psi=phi[j])
                filters.append(kern)
    return np.array(filters)


class Block2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),  # kernel size not in the paper
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = F.relu(self.basic(x) + self.shortcut(x))
        return x


class Block3(nn.Module):
    def __init__(self, in_channels: int = 32, out_channels: int = 64, groups: int = 32):
        super().__init__()
        if out_channels % groups:
            raise ValueError(f'out_channels {out_channels} not divisible by groups {groups}')
        self.basic = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, stride=2, groups=groups, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = F.relu(self.basic(x) + self.shortcut(x))
        return x


class Preprocessing(nn.Module):
    """Fixed SRM and Gabor filters on each channel, followed by TLU."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # fixed filters, follow .to(), predefined unless in the checkpoint
        filters = torch.from_numpy(build_filters()).view(62, 1, 5, 5).repeat(in_channels, 1, 1, 1)
        self.register_buffer('filters', filters)
        self.register_load_state_dict_pre_hook(self._rename_legacy)
        self.register_load_state_dict_pre_hook(_common.keep_missing_buffers)
        self.tlu = nn.Hardtanh(min_val=-2, max_val=2)

    @staticmethod
    def _rename_legacy(module, state_dict, prefix, *args):
        """Filters were stored as frozen convolution before."""
        if f'{prefix}0.weight' in state_dict:
            state_dict.setdefault(f'{prefix}filters', state_dict.pop(f'{prefix}0.weight'))

    def forward(self, x):
        x = F.conv2d(x, self.filters, padding=2, groups=x.size(1))
        return self.tlu(x)


class UcNet(nn.Module):
    """A convolutional neural network for steganalysis.
    Intended for binary classification of stego and cover color images.

    Introduced in
    K. Wei, W. Luo, S. Tan, J. Huang.
    Universal deep network for steganalysis of color image based on channel representation.
    IEEE Transactions on Information Forensics and Security, 2022.

    :param in_channels: number of channels of the input images, defaults to 3 (color)
    :type in_channels: int
    :param num_classes: number of output classes, defaults to 2 (one-hot binary)
    :type num_classes: int
    :param weight_seed: seed of the random initial weights,
        by default the global random number generator
    :type weight_seed: int
    :return: randomly initialized UcNet instance, to be trained or loaded with pretrained weights
    :rtype: UcNet

    :Example:

    >>> model = sw.nn.UcNet(in_channels=3, num_classes=2)
    """

    PRETRAINED_URL = 'https://github.com/revere7/UCNet_Steganalysis/raw/refs/heads/main/J-UNIWARD-pretrain-parameters.pt'

    def __init__(self, in_channels: int = 3, num_classes: int = 2, *, weight_seed: int = None):
        super().__init__()
        with _common.seeded(weight_seed):

            # === preprocessing module ===
            self.preprocessing = Preprocessing(in_channels)

            # === convolutional module ===
            self.block_1a = nn.Sequential(
                nn.Conv2d(in_channels*62, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
            self.block_2a = Block2(32, 32)
            self.block_3 = Block3(32, 64)
            self.block_2b = Block2(64, 128)
            self.block_1b = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )

            # === classification module ===
            self.gap = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
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
        ... ).permute(2, 0, 1)[None]  # [B, C, H, W]
        >>> logits = model(x)  # [B, 2]
        >>> y = torch.softmax(logits, dim=1)[:, 1]  # stego probability
        """
        # preprecessing module
        x = self.preprocessing(x)
        # convolutional module
        x = self.block_1a(x)
        x = self.block_2a(x)
        x = self.block_3(x)
        x = self.block_2b(x)
        x = self.block_1b(x)
        # classification module
        x = self.gap(x)
        y = self.fc(x)
        return y

    @classmethod
    def pretrained(
        cls,
        path: Union[str, Path] = None,
        *,
        in_channels: int = 3,
        device: torch.device = torch.device('cpu'),
        strict: bool = True,
        reset_classifier: bool = False,
        weight_seed: int = None,
        **kw,
    ) -> 'UcNet':
        """Loads pretrained model of the authors. Downloads if missing.

        The default weights are trained against J-UNIWARD.

        :param path: local path to the weights in the authors' format,
            by default downloads them into the torch hub cache
        :type path: str or pathlib.Path
        :param in_channels: number of input channels
        :type in_channels: int
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
        :rtype: UcNet

        :Example:

        >>> model = sw.nn.UcNet.pretrained()  # trained against J-UNIWARD
        >>> model = sw.nn.UcNet.pretrained("ucnet.pt")  # weights in the authors' format
        >>> y = sw.nn.infere_single(x0, model)  # stego probability
        """
        model = cls(in_channels=in_channels, weight_seed=weight_seed, **kw)
        source = cls.PRETRAINED_URL if path is None else path
        state_dict = _common.load_state_dict(source)['original_state']
        state_dict = convert_state_dict(state_dict, in_channels=in_channels)

        # keep FC weights of the random initialization
        if reset_classifier:
            state_dict['fc.weight'] = model.fc.weight.data
            state_dict['fc.bias'] = model.fc.bias.data
        model.load_state_dict(state_dict, strict=strict)
        logging.info(f'using pretrained weights from {source}')
        return model.to(device).eval()


def convert_state_dict(state_dict, *, in_channels: int = 3):
    """Converts state dict of the authors' implementation.

    https://github.com/revere7/UCNet_Steganalysis/

    :param state_dict: state dict of the authors' model
    :type state_dict: dict
    :param in_channels: number of input channels, the filters are applied to each
    :type in_channels: int
    :return: state dict of UcNet
    :rtype: dict
    """
    prefixes = {
        'group1.hpf.weight': 'preprocessing.filters',
        'group2.0.': 'block_1a.0.',
        'group2.1.': 'block_1a.1.',
        'group2.3.': 'block_1a.3.',
        'group2.4.': 'block_1a.4.',
        'group2.6.': 'block_2a.',
        'group3.': 'block_3.',
        'group4.': 'block_2b.',
        'group5.': 'block_1b.',
        'fc1.': 'fc.',
    }
    converted = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        for old, new in prefixes.items():
            if k.startswith(old):
                break
        else:
            raise KeyError(f'unexpected key {k}')
        if old == 'group1.hpf.weight':
            v = v.repeat(in_channels, 1, 1, 1)
        converted[new + k[len(old):]] = v
    return converted


def pretrained(*args, **kw) -> UcNet:
    return UcNet.pretrained(*args, **kw)


def infere_single(
    x: np.ndarray,
    model: UcNet = None,
    *,
    device: torch.device = None,
) -> np.ndarray:
    """Runs inference for a single image.

    :param x: color image of shape [H, W, 3] in [0, 255]
    :type x: np.ndarray
    :param model: model in eval mode,
        by default loads :func:`UcNet.pretrained` (pass the model for repeated calls)
    :type model: UcNet
    :param device: torch device to move the model to,
        by default the model stays where it is
    :type device: torch.device
    :return: stego probability
    :rtype: np.ndarray
    """
    if model is None:
        model = UcNet.pretrained()
    return _common.infere_single(x, model, device=device)
