"""

Author: Martin Benes
"""

import numpy as np
import torch

#
from . import _efficientnet
#
from ._efficientnet import EfficientNet, B0, B4
from .. import _common


def infere_single(
    x: np.ndarray,
    model: EfficientNet,
    *,
    device: torch.device = None,
) -> np.ndarray:
    """Runs inference for a single image.

    The model must be trained; the classifier of
    :func:`EfficientNet.pretrained` without path is untrained.

    :param x: grayscale image of shape [H, W], or color image of shape [H, W, C],
        matching the in_channels of the model
    :type x: np.ndarray
    :param model: trained model in eval mode
    :type model: EfficientNet
    :param device: torch device to move the model to,
        by default the model stays where it is
    :type device: torch.device
    :return: stego probability
    :rtype: np.ndarray
    """
    return _common.infere_single(x, model, device=device)


__all__ = [
    'EfficientNet',
    'B0',
    'B4',
    'infere_single',
]
