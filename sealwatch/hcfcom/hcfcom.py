"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from collections import OrderedDict
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union
import warnings


def _hcfcom(
    x: np.ndarray,
    *,
    order: int = 1,
) -> float:
    """HCF-COM of a single channel.

    As defined in A. D. Ker, Steganalysis of LSB matching in grayscale images,
    IEEE Signal Processing Letters 12(6), 2005, following Harmsen and Pearlman,

    C(H) = sum_{k=0}^{N/2} k^order |H[k]| / sum_{k=0}^{N/2} |H[k]|,

    where H is the DFT of the histogram, of N = 256 bins.

    :param x: single channel of uint8 values
    :type x: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param order: order of the center of mass
    :type order: int
    :return: center of mass of the histogram characteristic function
    :rtype: float
    """
    # histogram characteristic function (HCF)
    h, _ = np.histogram(x, range(257))
    # magnitudes up to N/2, the rest is redundant
    hcf = np.abs(np.fft.fft(h))[:129]
    # center of mass (COM)
    k = np.arange(129)
    return np.sum(k ** order * hcf) / np.sum(hcf)


def extract_from_file(
    path: Union[str, Path],
    **kw,
) -> OrderedDict:
    """Extract HCF-COM from an image file, decoded with PIL.

    :param path: grayscale or color image, e.g., PNG
    :type path: str or pathlib.Path
    :param kw: other parameters of extract()
    :return: HCF-COM per channel, see extract()
    :rtype: OrderedDict

    :Example:

    >>> features = sw.hcfcom.extract_from_file('seal1.png')
    """
    x = np.array(Image.open(path))
    return extract(x, **kw)


def extract(
    x: np.ndarray = None,
    *,
    order: int = 1,
    x1: np.ndarray = None,
) -> OrderedDict:
    """Extract the center of mass of the histogram characteristic function (HCF-COM).

    Introduced in
    J. J. Harmsen and W. A. Pearlman, Steganalysis of additive noise modelable information hiding,
    Proc. SPIE, Security and Watermarking of Multimedia Contents V, vol. 5020, 2003.

    :param x: grayscale image of shape [height, width], or color image of shape [height, width, channels],
        of uint8 values
    :type x: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param order: order of the center of mass
    :type order: int
    :param x1: deprecated, use x
    :type x1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: HCF-COM per channel, the keys are the channel indices
    :rtype: `OrderedDict`

    :Example:

    >>> features = sw.hcfcom.extract(x)
    """
    if x1 is not None:
        warnings.warn('parameter x1 is deprecated, use x', DeprecationWarning, stacklevel=2)
        if x is not None:
            raise TypeError('both x and x1 given')
        x = x1
    if x is None:
        raise TypeError('x must be given')

    # expand grayscale
    if len(x.shape) == 2:
        x = x[..., None]

    # calculate HCF-COM per channel
    return OrderedDict([
        (str(ch), _hcfcom(x[..., ch], order=order))
        for ch in range(x.shape[-1])
    ])
