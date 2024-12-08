"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import jpeglib
import numpy as np
from pathlib import Path
from typing import Union

from .dct import block_idct2, jpeglib_to_jpegio
# from sealwatch.utils.dct import block_idct2


def decompress(
    y: np.ndarray,
    qt: np.ndarray = None,
    dct_mat: np.ndarray = None,
) -> np.ndarray:
    """Decompresses a DCT without final rounding.

    :param y: quantized DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param dct_mat: pre-computed DCT matrix. If None, the DCT matrix will be computed on the fly.
    :type dct_mat: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: decompressed channel of shape [num_vertical_blocks * 8, num_horizontal_blocks * 8]. The values should be in the range [0, 255], although some values may exceed this range.
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    num_vertical_blocks, num_horizontal_blocks = y.shape[:2]

    # If quantization table is given, unquantize the DCT coefficients
    if qt is not None:
        assert qt.shape == (8, 8), 'expected quantization table of shape [8, 8]'
        y = y * qt[None, None, :, :]

    # Inverse DCT
    x = block_idct2(y, dct_mat=dct_mat)

    # Reorder blocks to obtain image
    x = jpeglib_to_jpegio(x)

    # Level shift
    x += 128

    return x


def decompress_from_file(
    path: Union[str, Path],
    dct_mat: np.ndarray = None,
) -> np.ndarray:
    """Decompresses luminance from given JPEG without final rounding.

    :param path: path to JPEG image
    :type path: pathlib.Path or str
    :param dct_mat: pre-computed DCT matrix. If None, will be computed on the fly.
    :type dct_mat:
    :return: decompressed luminance channel, ndarray of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    # read DCT
    jpeg = jpeglib.read_dct(path)
    qt = jpeg.get_component_qt(0)

    # decompress without rounding
    x = decompress(
        y=jpeg.Y,
        qt=qt,
        dct_mat=dct_mat,
    )

    # Crop pixels that exceed the image boundary
    if x.shape[0] - jpeg.height > 0:
        assert x.shape[0] - jpeg.height < 8, "Expected less than 8 pixels"
        x = x[:jpeg.height]
    if x.shape[1] - jpeg.width > 0:
        assert x.shape[1] - jpeg.width < 8, "Expected less than 8 pixels"
        x = x[:, :jpeg.width]

    return x
