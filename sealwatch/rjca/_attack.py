"""Module performing reverse JPEG compatibility attack.

Introduced in
J. Butora, Jessica Fridrich
"Reverse JPEG Compatibility Attack"
IEEE TIFS, 2019.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import warnings

from .. import tools


def attack(
    y1: np.ndarray,
    qt: np.ndarray,
) -> float:
    """Performs RJCA and returns variance.

    Rounding error should be around 0.04-0.07.
    For stego, it grows towards 0.08333 (1/12).

    :param y1: quantized cover DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table
        of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: variance of the rounding error
    :rtype: float

    :Example:

    >>> jpeg = jpeglib.read_dct('suspicious.jpeg')
    >>> var = cl.rjca.attack(
    ... 	y1=jpeg.Y,
    ...		qt=jpeg.qt[0],
	... )
    >>> assert np.abs(var - 1/12.) > .005
    """
    # check QT
    if (qt != 1).any():
        warnings.warn('running RJCA on <QF100 does not work')

    # estimate rounding error
    spatial = tools.dct.block_idct2(y1 * qt[None, None])
    err = spatial - np.round(spatial)

    # variance
    return np.var(err)
