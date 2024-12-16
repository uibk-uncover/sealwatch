"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np


def compute_dct_mat():
    """
    Computes the 8x8 DCT matrix
    :return: ndarray of shape [8, 8]
    """
    #
    [col, row] = np.meshgrid(range(8), range(8))
    dct_mat = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))
    dct_mat[0, :] = dct_mat[0, :] / np.sqrt(2)
    return dct_mat


def block_dct2(x, dct_mat=None):
    """Apply 2D DCT to image blocks.

    :param spatial_blocks: ndarray of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :param dct_mat: ndarray of shape [8, 8]. If None, the DCT matrix is computed on the fly.
    :return: DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    """
    # compute DCT matrix
    if dct_mat is None:
        dct_mat = compute_dct_mat()
    #
    dct_mat_left = dct_mat[None, None, :, :]
    dct_mat_right = (dct_mat.T)[None, None, :, :]

    #
    y = dct_mat_left @ x @ dct_mat_right
    return y


def block_idct2(
    y: np.ndarray,
    *,
    dct_mat: np.ndarray = None,
) -> np.ndarray:
    """
    Apply 2D inverse DCT to image blocks
    :param y: DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param dct_mat: DCT matrix
        of shape [8, 8].
        By default, it is computed on the fly. Repeated calls can be providing a pre-calculated DCT matrix.
    :return: spatial blocks of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    """
    # compute DCT matrix
    if dct_mat is None:
        dct_mat = compute_dct_mat()
    #
    dct_mat_left = (dct_mat.T)[None, None, :, :]
    dct_mat_right = dct_mat[None, None, :, :]

    # iDCT transform as matrix product
    x = dct_mat_left @ y @ dct_mat_right
    return x


def jpeglib_to_jpegio(
    y: np.ndarray,
) -> np.ndarray:
    """
    Convert DCT coefficient array shape from 4D jpeglib format to 2D jpegio format

    :param y: DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :param y: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: DCT coefficients
        of shape [num_vertical_blocks * 8, num_horizontal_blocks * 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    num_vertical_blocks, num_horizontal_blocks, block_height, block_width = y.shape
    assert block_height == 8, "Expected block height of 8"
    assert block_width == 8, "Expected block width of 8"

    # Transpose from [num_vertical_blocks, num_horizontal_blocks, 8, 8] to [num_vertical_blocks, 8, num_horizontal_blocks, 8]
    y = y.transpose((0, 2, 1, 3))

    # Reshape to [num_vertical_blocks * 8, num_horizontal_blocks * 8]
    y = y.reshape((num_vertical_blocks * 8, num_horizontal_blocks * 8))
    return y


def jpegio_to_jpeglib(y: np.ndarray) -> np.ndarray:
    """
    Convert DCT coefficient array shape from 2D jpegio format to 4D jpeglib format
    :param y: DCT coefficients
        of shape [num_vertical_blocks * 8, num_horizontal_blocks * 8]
    :type y: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    # From jpegio 2D to jpeglib 4D
    assert y.shape[0] % 8 == 0
    assert y.shape[1] % 8 == 0

    num_vertical_blocks = y.shape[0] // 8
    num_horizontal_blocks = y.shape[1] // 8

    # Reshape from [num_vertical_blocks * 8, num_horizontal_blocks * 8] to [num_vertical_blocks, 8, num_horizontal_blocks, 8]
    y = y.reshape((num_vertical_blocks, 8, num_horizontal_blocks, 8))

    # Reorer to [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    y = y.transpose((0, 2, 1, 3))

    return y
