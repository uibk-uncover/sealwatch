"""Implementation of SRNet.

Introduced in
M. Boroumand, M. Chen, J. Fridrich
Deep residual network for steganalysis of digital images
IEEE TIFS, 2019.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import logging
import numpy as np
from pathlib import Path
import torch
from torch import nn, Tensor
from typing import Union

from .. import _common


class Type1(nn.Module):
    """Layer type 1: 3x3 convolution, BN, ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Returns conv -> BN -> ReLU."""
        return self.activation(self.norm(self.conv(x)))


class Type2(nn.Module):
    """Layer type 2: type 1, 3x3 convolution, BN, identity shortcut.

    No ReLU after the shortcut (Sec. II-B1).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layer1 = Type1(in_channels, out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Returns type 1 -> conv -> BN, plus input."""
        return self.norm(self.conv(self.layer1(x))) + x


class Type3(nn.Module):
    """Layer type 3: type 1, 3x3 convolution, BN, 3x3 average pooling,
    shortcut with 1x1 convolution with stride 2 and BN."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layer1 = Type1(in_channels, out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.resnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Returns type 1 -> conv -> BN -> pooling, plus shortcut."""
        y = self.pool(self.norm(self.conv(self.layer1(x))))
        return y + self.resnorm(self.resconv(x))


class Type4(nn.Module):
    """Layer type 4: type 1, 3x3 convolution, BN, global average pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layer1 = Type1(in_channels, out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Returns type 1 -> conv -> BN -> global pooling."""
        return self.pool(self.norm(self.conv(self.layer1(x))))


class SRNet(nn.Module):
    """
    SRNet: A deep residual network for steganalysis.

    Architecture (Fig. 1):
    - 2 layers of type 1, extracting noise residuals, no pooling
    - 5 layers of type 2 with identity shortcuts, no pooling
    - 4 layers of type 3 with pooling and convolutional shortcuts
    - 1 layer of type 4 with global average pooling
    - Fully connected layer

    Intended for binary classification of stego and cover images.
    """
    PRETRAINED_URL = 'https://github.com/DDELab/deepsteganalysis/releases/download/v0.1/jin_srnet.ckpt'

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        # layers 1-2
        self.block1 = nn.Sequential(
            Type1(in_channels, 64),
            Type1(64, 16),
        )
        # layers 3-7
        self.block2 = nn.Sequential(*[Type2(16, 16) for _ in range(5)])
        # layers 8-11
        self.block3 = nn.Sequential(
            Type3(16, 16),
            Type3(16, 64),
            Type3(64, 128),
            Type3(128, 256),
        )
        # layer 12
        self.block4 = nn.Sequential(
            Type4(256, 512),
        )
        # classification
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """He initialization of the convolutions, Gaussian of the FC (Sec. III-B)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """Returns the feature vector of layer 12."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass, returns logits."""
        return self.fc(self.forward_features(x))

    @classmethod
    def pretrained(
        cls,
        path: Union[str, Path] = None,
        *,
        in_channels: int = 1,
        device: torch.device = torch.device('cpu'),
        strict: bool = True,
        reset_classifier: bool = False,
        seed: int = None,
        **kw,
    ) -> 'SRNet':
        """Loads pretrained model. Downloads if missing.

        The default weights by Jin are from DDELab/deepsteganalysis,
        trained on color images.
        For grayscale, the green channel of the first convolution is taken.

        :param path: local path to a checkpoint,
            by default downloads Jin's weights into the torch hub cache
        :type path: str or pathlib.Path
        :param in_channels: number of input channels, 1 or 3
        :type in_channels: int
        :param device: torch device
        :type device: torch.device
        :param strict: whether the weights must match exactly
        :type strict: bool
        :param reset_classifier: whether to reinitialize the FC layer randomly
        :type reset_classifier: bool
        :param seed: seed for the reinitialization of the FC layer
        :type seed: int
        :return: loaded model in eval mode
        :rtype: SRNet
        """
        model = cls(in_channels=in_channels, **kw)
        source = cls.PRETRAINED_URL if path is None else path
        state_dict = _common.load_state_dict(source)
        state_dict = state_dict.get('state_dict', state_dict)

        # strip wrapper prefixes of Lightning (net.) and DataParallel (module.)
        def strip(k):
            for prefix in ('net.', 'module.'):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            return k
        state_dict = {strip(k): v for k, v in state_dict.items()}

        # modify input
        stem = state_dict['block1.0.conv.weight']
        if stem.size(1) == 3 and in_channels == 1:
            state_dict['block1.0.conv.weight'] = stem[:, 1:2]
        elif stem.size(1) != in_channels:
            raise NotImplementedError(f'undefined initialization of {in_channels} channels from {stem.size(1)}')

        # initialize FC weights randomly
        if reset_classifier:
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed)
            state_dict['fc.weight'] = torch.empty_like(model.fc.weight.data).normal_(generator=generator)
            state_dict['fc.bias'] = torch.empty_like(model.fc.bias.data).normal_(generator=generator)

        #
        model.load_state_dict(state_dict, strict=strict)
        logging.info(f'using pretrained weights from {source}')
        return model.to(device).eval()


def pretrained(*args, **kw) -> SRNet:
    return SRNet.pretrained(*args, **kw)


def infere_single(
    x: np.ndarray,
    model: SRNet = None,
    *,
    device: torch.device = None,
) -> np.ndarray:
    """Runs inference for a single image.

    :param x: grayscale image of shape [H, W] in [0, 255],
        or color image of shape [H, W, 3] for in_channels=3
    :type x: np.ndarray
    :param model: model in eval mode,
        by default loads :func:`SRNet.pretrained` (pass the model for repeated calls)
    :type model: SRNet
    :param device: torch device to move the model to,
        by default the model stays where it is
    :type device: torch.device
    :return: stego probability
    :rtype: np.ndarray
    """
    if model is None:
        model = SRNet.pretrained()
    return _common.infere_single(x, model, device=device)
