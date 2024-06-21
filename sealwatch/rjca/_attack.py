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

from .. import utils


def attack(
    dct_coeffs: np.ndarray,
    quantization_table: np.ndarray,
) -> float:
    """Performs RJCA and returns variance.

    Rounding error should be around 0.04-0.07.
    For stego, it grows towards 0.08333 (1/12).

    :param dct_coeffs: quantized cover DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type dct_coeffs: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param quantization_table: quantization table
        of shape [8, 8]
    :type quantization_table: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: variance of the rounding error
    :rtype: float

    :Example:
    >>> jpeg = jpeglib.read_dct('suspicious.jpeg')
    >>> var = cl.rjca.attack(
    ... 	dct_coeffs=jpeg.Y,
    ...		quantization_table=jpeg.qt[0],
	... )
    >>> assert np.abs(var - 1/12.) > .005
    """
    # check QT
    if (quantization_table != 1).any():
        warnings.warn('running RJCA on <QF100 does not work')

    # estimate rounding error
    spatial = utils.dct.block_idct2(
        dct_coeffs * quantization_table[None, None]
    )
    err = spatial - np.round(spatial)

    # variance
    return np.var(err)
