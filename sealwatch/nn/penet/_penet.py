"""
Implementation of PENet, as described in

K. Wei, W. Luo, and J. Huang, Color Image Steganalysis Based on Pixel Difference Convolution
and Enhanced Transformer With Selective Pooling.
IEEE Transactions on Information Forensics and Security, vol. 19, pp. 9970-9983, 2024.
https://doi.org/10.1109/TIFS.2024.3486027

The network for arbitrary-size images, with spatial pyramid pooling.
The authors' implementation is https://github.com/revere7/PENet_Steganalysis

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import enum
import logging
import math
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

from .. import _common
from ..ucnet._ucnet import build_filters


class Implementation(enum.Enum):
    """PENet implementation to choose from."""

    PENET_ORIGINAL = enum.auto()
    """As the authors' implementation.

    The LayerNorm of the queries normalizes over the whole batch, so that the prediction
    of an image depends on the other images in the batch, and the output of the attention
    is reshaped to the feature map without transposing the tokens and the channels.
    """
    PENET_FIX = enum.auto()
    """As described in the paper, with LayerNorm of each token (Eq. 6),
    and the output of the attention transposed back to the feature map."""


class PDConv(torch.nn.Module):
    """Pixel difference convolution, Eq. 2-5 of the paper.

    3x3 convolution with the weights w' of the 8 neighbors and zero in the center,
    combined with the difference to the center pixel, weighted by eta,

    y''(p0) = sum_n w'(pn) m(p0 + pn) - eta * sum_n w'(pn) m(p0).
    """
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, eta: float = .5):
        super().__init__()
        # weights of the 8 neighbors, in row-major order
        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, 8))
        self.stride = stride
        self.eta = eta

    def forward(self, x):
        # 3x3 kernel with zero center
        center = self.weight.new_zeros(self.weight.shape[:2] + (1,))
        kernel = torch.cat([self.weight[..., :4], center, self.weight[..., 4:]], dim=2)
        kernel = kernel.view(self.weight.shape[:2] + (3, 3))
        y = F.conv2d(x, kernel, stride=self.stride, padding=1)
        # difference to the center pixel, as 1x1 convolution with the summed weights
        y_center = F.conv2d(x, self.weight.sum(2)[..., None, None], stride=self.stride)
        return y - self.eta * y_center


class Stem(torch.nn.Module):
    """Fixed 30 SRM and 32 Gabor filters, applied on each color channel separately,
    concatenated, and truncated (TLU)."""
    def __init__(self, threshold: float = 5.):
        super().__init__()
        # fixed filters, follow .to(), predefined unless in the checkpoint
        self.register_buffer('filters', torch.from_numpy(build_filters()).view(62, 1, 5, 5))
        self.register_load_state_dict_pre_hook(_common.keep_missing_buffers)
        self.threshold = threshold

    def forward(self, x):
        b, c, h, w = x.shape
        # same filters on each channel, the residuals of a channel next to each other
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.filters, padding=2)
        x = x.view(b, c * 62, h, w)
        return torch.clamp(x, min=-self.threshold, max=self.threshold)


def pdconv_block(in_channels: int, out_channels: int, *, stride: int = 1, eta: float = .5) -> List[torch.nn.Module]:
    """PDConv, batch normalization and ReLU."""
    return [
        PDConv(in_channels, out_channels, stride=stride, eta=eta),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(),
    ]


def output_size(size: int) -> int:
    """Size after a convolution or pooling with kernel 3, stride 2, and padding 1."""
    return (size - 1) // 2 + 1


class EnhancedTransformer(torch.nn.Module):
    """Enhanced Transformer, Eq. 6-8 of the paper.

    Queries of all tokens, keys and values downsampled by a PDConv with stride 2,
    and a learned relative position bias. The input is added to the output.
    """
    def __init__(
        self,
        channels: int,
        tokens: Tuple[int, int],
        *,
        heads: int = 2,
        d_k: int = 64,
        eta: float = .5,
        implementation: Implementation = Implementation.PENET_FIX,
    ):
        super().__init__()
        self.heads, self.d_k = heads, d_k
        self.implementation = implementation
        # branch 1: queries
        if implementation is Implementation.PENET_FIX:
            self.norm = torch.nn.LayerNorm(channels)
        self.fc_q = torch.nn.Linear(channels, heads * d_k)
        # branch 2: keys and values
        self.conv_k = PDConv(channels, channels, stride=2, eta=eta)
        self.conv_v = PDConv(channels, channels, stride=2, eta=eta)
        self.fc_k = torch.nn.Linear(channels, heads * d_k)
        self.fc_v = torch.nn.Linear(channels, heads * d_k)
        self.fc_o = torch.nn.Linear(heads * d_k, channels)
        # relative position bias sigma, between each query and each key
        h, w = tokens
        self.position_bias = torch.nn.Parameter(torch.empty(1, heads, h * w, output_size(h) * output_size(w)))

    def _tokens(self, x: torch.Tensor, fc: torch.nn.Linear) -> torch.Tensor:
        """Tokens of the feature map, projected and split into heads, of shape [b, heads, tokens, d_k]."""
        b, c, h, w = x.shape
        x = fc(x.view(b, c, h * w).transpose(1, 2))
        return x.view(b, h * w, self.heads, self.d_k).transpose(1, 2)

    def forward(self, x):
        b, c, h, w = x.shape
        # queries, Eq. 6
        tokens = x.view(b, c, h * w).transpose(1, 2)
        if self.implementation is Implementation.PENET_ORIGINAL:
            tokens = F.layer_norm(tokens, tokens.shape)  # over the whole batch
        else:
            tokens = self.norm(tokens)
        q = self.fc_q(tokens).view(b, h * w, self.heads, self.d_k).transpose(1, 2)
        # keys and values, Eq. 7
        k = self._tokens(self.conv_k(x), self.fc_k)
        v = self._tokens(self.conv_v(x), self.fc_v)
        # attention, Eq. 8
        attn = q @ k.transpose(2, 3) * self.d_k ** -.5 + self.position_bias
        y = torch.softmax(attn, dim=-1) @ v
        y = self.fc_o(y.transpose(1, 2).reshape(b, h * w, self.heads * self.d_k))
        if self.implementation is Implementation.PENET_ORIGINAL:
            y = y.reshape(b, c, h, w)  # tokens and channels not transposed
        else:
            y = y.transpose(1, 2).reshape(b, c, h, w)
        return y + x


class SpatialPyramidPooling(torch.nn.Module):
    """Average pooling to grids of 4x4, 2x2, and 1x1, concatenated, Eq. 9 of the paper."""
    def __init__(
        self,
        grids: Tuple[int] = (4, 2, 1),
        *,
        implementation: Implementation = Implementation.PENET_FIX,
    ):
        super().__init__()
        self.grids = grids
        self.implementation = implementation

    def forward(self, x):
        b, c, h, w = x.shape
        pooled = []
        for n in self.grids:
            if self.implementation is Implementation.PENET_ORIGINAL:
                # window rounded up, stride rounded down, gives n x n for most sizes
                y = F.avg_pool2d(x, (math.ceil(h / n), math.ceil(w / n)), stride=(h // n, w // n))
            else:
                y = F.adaptive_avg_pool2d(x, n)
            pooled.append(y.reshape(b, -1))
        return torch.cat(pooled, dim=1)


class PENet(torch.nn.Module):
    """A convolutional neural network with a transformer for steganalysis.
    Intended for binary classification of stego and cover color images, of the given size.

    The relative position bias of the attention is learned for the size,
    so that a trained model applies to images of this size only.

    Introduced in
    K. Wei, W. Luo, J. Huang.
    Color image steganalysis based on pixel difference convolution and enhanced transformer with selective pooling.
    IEEE Transactions on Information Forensics and Security, 2024.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        input_size: Union[int, Tuple[int, int]] = 512,
        *,
        implementation: Implementation = Implementation.PENET_FIX,
        eta: float = .5,
        head: str = 'paper',
        weight_seed: int = None,
    ):
        """
        :param in_channels: number of input channels
        :type in_channels: int
        :param num_classes: number of output classes
        :type num_classes: int
        :param input_size: height and width of the input images, 512 in the paper
        :type input_size: int or tuple
        :param implementation: implementation of PENet, sw.PENET_FIX or sw.PENET_ORIGINAL
        :type implementation: `Implementation`
        :param eta: weight of the pixel difference in PDConv, Eq. 5 of the paper,
            0.5 in the authors' implementation
        :type eta: float
        :param head: classification module, 'paper' with two fully-connected layers (Table I),
            or 'linear' with one. As there is no activation between the two layers of the paper,
            both represent the same linear functions, but 'linear' has 5.5M parameters less.
        :type head: str
        :param weight_seed: seed of the random initial weights,
            by default the global random number generator
        :type weight_seed: int
        :return: randomly initialized PENet instance, to be trained or loaded with trained weights
        :rtype: PENet

        :Example:

        >>> model = sw.nn.PENet(in_channels=3, num_classes=2, input_size=512)
        """
        super().__init__()
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)

        # stem module
        self.stem = Stem()

        # pixel difference convolution module, Table I, stride 2 for arbitrary-size images
        self.block1 = torch.nn.Sequential(
            *pdconv_block(in_channels * 62, 64, stride=2, eta=eta),
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block2 = torch.nn.Sequential(
            *pdconv_block(64, 128, eta=eta),
            *pdconv_block(128, 128, eta=eta),
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block3 = torch.nn.Sequential(
            *pdconv_block(128, 256, eta=eta),
            *pdconv_block(256, 256, eta=eta),
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        # enhanced Transformer module with selective pooling
        tokens = self.input_size
        for _ in range(4):  # strided PDConv and three pooling layers
            tokens = tuple(output_size(s) for s in tokens)
        self.transformer = EnhancedTransformer(256, tokens, eta=eta, implementation=implementation)
        self.spp = SpatialPyramidPooling(implementation=implementation)

        # classification module
        self.head = head
        if head == 'paper':
            # Table I
            self.fc1 = torch.nn.Linear((16 + 4 + 1) * 256, 1024)
            self.fc2 = torch.nn.Linear(1024, num_classes)
        elif head == 'linear':
            self.fc = torch.nn.Linear((16 + 4 + 1) * 256, num_classes)
        else:
            raise ValueError(f'unknown head {head}, expected paper or linear')

        # initial weights
        with torch.random.fork_rng():
            if weight_seed is not None:
                torch.manual_seed(weight_seed)
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: torch.nn.Module):
        """Initial weights of the authors' implementation, He initialization of the convolutions."""
        if isinstance(module, PDConv):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=.01)
            # PyTorch default, as the authors' implementation
            bound = 1 / math.sqrt(module.in_features)
            torch.nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, EnhancedTransformer):
            # not initialized in the authors' implementation, randomly in the paper
            torch.nn.init.trunc_normal_(module.position_bias, std=.02)

    def forward(self, x):
        """Forward pass, returns logits.

        :param x: images of shape [B, C, H, W] in [0, 1],
            where B is the batch size, C is the number of channels (in_channels),
            and H and W are the height and width of the images (input_size)
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
        if tuple(x.shape[2:]) != self.input_size:
            raise ValueError(f'expected images of size {self.input_size}, got {tuple(x.shape[2:])}')
        # the stem filters and threshold are for [0, 255], as of the authors
        x = self.stem(x * 255.)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.transformer(x)
        x = self.spp(x)
        if self.head == 'paper':
            return self.fc2(self.fc1(x))
        else:
            return self.fc(x)

    @classmethod
    def pretrained(
        cls,
        path: Union[str, Path] = None,
        *,
        in_channels: int = 3,
        device: torch.device = torch.device('cpu'),
        strict: bool = True,
        **kw,
    ) -> 'PENet':
        """Loads a trained model.

        The authors have not published trained weights, so that the path must be given.
        Checkpoints of the authors' implementation (PENet_Arbitrary.py) are converted,
        and need implementation=PENET_ORIGINAL.
        The two fully-connected layers of the paper are merged into one for head='linear',
        with the same predictions, see merge_head().

        :param path: local path to a checkpoint of PENet, or of the authors' implementation
        :type path: str or pathlib.Path
        :param in_channels: number of input channels
        :type in_channels: int
        :param device: torch device
        :type device: torch.device
        :param strict: whether the weights must match exactly
        :type strict: bool
        :return: loaded model in eval mode
        :rtype: PENet

        :Example:

        >>> model = sw.nn.PENet.pretrained("penet.pt", input_size=512)  # no published weights
        >>> y = sw.nn.infere_single(x0, model)  # stego probability
        """
        if path is None:
            raise ValueError('no trained weights of PENet are published, give the path to a checkpoint')
        model = cls(in_channels=in_channels, **kw)
        state_dict = _common.load_state_dict(path)
        # checkpoints of the authors' training
        state_dict = state_dict.get('original_state', state_dict)
        state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        if any(k.startswith(('group1.', 'entrans.')) for k in state_dict):
            state_dict = convert_state_dict(state_dict)

        # head
        if model.head == 'linear' and 'fc1.weight' in state_dict:
            state_dict = merge_head(state_dict)
        elif model.head == 'paper' and 'fc.weight' in state_dict:
            raise ValueError('checkpoint has the linear head, use head=\'linear\'')

        model.load_state_dict(state_dict, strict=strict)
        logging.info(f'using pretrained weights from {path}')
        return model.to(device).eval()


def merge_head(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Merges the two fully-connected layers of the paper's head into one, for head='linear'.

    As there is no activation between them, fc2(fc1(x)) = W x + b,
    with W = W2 W1 and b = W2 b1 + b2, so that the predictions do not change.
    The product is computed in double precision.

    :param state_dict: state dict of PENet with head='paper'
    :type state_dict: dict
    :return: state dict of PENet with head='linear'
    :rtype: dict
    """
    state_dict = dict(state_dict)
    w1, b1 = state_dict.pop('fc1.weight'), state_dict.pop('fc1.bias')
    w2, b2 = state_dict.pop('fc2.weight'), state_dict.pop('fc2.bias')
    state_dict['fc.weight'] = (w2.double() @ w1.double()).to(w1.dtype)
    state_dict['fc.bias'] = (w2.double() @ b1.double() + b2.double()).to(b1.dtype)
    return state_dict


def convert_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Converts a state dict of the authors' implementation (PENet_Arbitrary.py),
    for PENet with PENET_ORIGINAL.

    :param state_dict: state dict of the authors' Net
    :type state_dict: dict
    :return: state dict of PENet
    :rtype: dict
    """
    names = {
        'group1.hpf.weight': 'stem.filters',
        'group2.basic.0.conv.weight': 'block1.0.weight',
        'group2.basic.1.': 'block1.1.',
        'group3.basic.0.conv.weight': 'block2.0.weight',
        'group3.basic.1.': 'block2.1.',
        'group3.basic.3.conv.weight': 'block2.3.weight',
        'group3.basic.4.': 'block2.4.',
        'group4.basic.0.conv.weight': 'block3.0.weight',
        'group4.basic.1.': 'block3.1.',
        'group4.basic.3.conv.weight': 'block3.3.weight',
        'group4.basic.4.': 'block3.4.',
        'entrans.B': 'transformer.position_bias',
        'entrans.dwconv_k.down.conv.weight': 'transformer.conv_k.weight',
        'entrans.dwconv_v.down.conv.weight': 'transformer.conv_v.weight',
        'entrans.': 'transformer.',
        'fc1.': 'fc1.',
        'fc2.': 'fc2.',
    }
    converted = {}
    for key, value in state_dict.items():
        # without DataParallel prefix
        if key.startswith('module.'):
            key = key[len('module.'):]
        old = next((old for old in names if key == old or (old.endswith('.') and key.startswith(old))), None)
        if old is None:
            raise KeyError(f'unexpected key {key}')
        new = names[old] + key[len(old):]
        if value.ndim == 4 and value.shape[2:] == (1, 8):
            value = value.reshape(value.shape[:2] + (8,))  # PDConv weights
        converted[new] = value
    return converted


def pretrained(*args, **kw) -> PENet:
    return PENet.pretrained(*args, **kw)


def infere_single(
    x: np.ndarray,
    model: PENet,
    *,
    device: torch.device = None,
) -> np.ndarray:
    """Runs inference for a single image.

    :param x: color image of shape [H, W, 3] in [0, 255], of the input size of the model
    :type x: np.ndarray
    :param model: trained model in eval mode
    :type model: PENet
    :param device: torch device to move the model to,
        by default the model stays where it is
    :type device: torch.device
    :return: stego probability
    :rtype: np.ndarray
    """
    return _common.infere_single(x, model, device=device)
