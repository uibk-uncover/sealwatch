"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from collections import OrderedDict
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


def extract(
    x1: np.ndarray,
    *,
    order: int = 1,
) -> np.ndarray:
    """

    :param x:
    :type x: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param order:
    :type order:
    :return:
    :rtype: `OrderedDict`

    :Example:

    >>> # TODO
    """
    # expand grayscale
    if len(x1.shape) == 2:
        x1 = x1[..., None]

    # calculate HCF-COM per channel
    return OrderedDict([
        (str(ch), _hcfcom(x1[..., ch], order=1))
        for ch in range(x1.shape[-1])
    ])
