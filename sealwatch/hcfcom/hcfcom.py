"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np


def _hcfcom(
    x: np.ndarray,
    *,
    order: int = 1,
) -> float:
    """

    :param x:
    :type x:
    :param order:
    :type order:
    :return:
    :rtype:
    """
    # histogram characteristic function (HCF)
    h, bins = np.histogram(x, range(257), density=True)
    hcf = np.fft.fft(h)
    # center of mass (COM)
    hcf = np.abs(hcf[1:129])  # remove DC and right half
    return sum([
        (k/256)**order * hcf[k]
        for k in range(128)
    ]) / np.sum(hcf)


def extract_hcfcom(
    x1: np.ndarray,
    *,
    order: int = 1,
) -> np.ndarray:
    """

    :param x:
    :type x:
    :param order:
    :type order:
    :return:
    :rtype:

    :Example:

    >>> # TODO
    """
    # expand grayscale
    if len(x1.shape) == 2:
        x1 = x1[..., None]

    # calculate HCF-COM per channel
    return np.array([
        _hcfcom(x1[..., ch], order=1)
        for ch in range(x1.shape[-1])
    ])
