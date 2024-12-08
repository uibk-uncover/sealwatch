import numpy as np


def compute_dct_mat():
    """
    Computes the 8x8 DCT matrix
    :return: ndarray of shape [8, 8]
    """
    [col, row] = np.meshgrid(range(8), range(8))
    dct_mat = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))
    dct_mat[0, :] = dct_mat[0, :] / np.sqrt(2)
    return dct_mat


def block_dct2(spatial_blocks, dct_mat=None):
    """
    Apply 2D DCT to image blocks
    :param spatial_blocks: ndarray of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :param dct_mat: ndarray of shape [8, 8]. If None, the DCT matrix is computed on the fly.
    :return: DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    """
    if dct_mat is None:
        dct_mat = compute_dct_mat()

    dct_mat_left = dct_mat[None, None, :, :]
    dct_mat_right = (dct_mat.T)[None, None, :, :]

    dct_coeffs = dct_mat_left @ spatial_blocks @ dct_mat_right

    return dct_coeffs


def block_idct2(dct_coeffs, dct_mat=None):
    """
    Apply 2D inverse DCT to image blocks
    :param dct_coeffs: ndarray of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :param dct_mat: ndarray of shape [8, 8]. If None, the DCT matrix is computed on the fly.
    :return: spatial blocks of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    """
    if dct_mat is None:
        dct_mat = compute_dct_mat()

    dct_mat_left = (dct_mat.T)[None, None, :, :]
    dct_mat_right = dct_mat[None, None, :, :]

    spatial_blocks = dct_mat_left @ dct_coeffs @ dct_mat_right

    return spatial_blocks


def jpeglib_to_jpegio(dct_coeffs: np.ndarray) -> np.ndarray:
    """
    Convert DCT coefficient array shape from 4D jpeglib format to 2D jpegio format
    :param dct_coeffs: DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :return: DCT coefficients reshaped to [num_vertical_blocks * 8, num_horizontal_blocks * 8]
    """
    num_vertical_blocks, num_horizontal_blocks, block_height, block_width = dct_coeffs.shape
    assert block_height == 8, "Expected block height of 8"
    assert block_width == 8, "Expected block width of 8"

    # Transpose from [num_vertical_blocks, num_horizontal_blocks, 8, 8] to [num_vertical_blocks, 8, num_horizontal_blocks, 8]
    dct_coeffs = dct_coeffs.transpose((0, 2, 1, 3))

    # Reshape to [num_vertical_blocks * 8, num_horizontal_blocks * 8]
    dct_coeffs = dct_coeffs.reshape((num_vertical_blocks * 8, num_horizontal_blocks * 8))

    return dct_coeffs


def jpegio_to_jpeglib(dct_coeffs: np.ndarray) -> np.ndarray:
    """
    Convert DCT coefficient array shape from 2D jpegio format to 4D jpeglib format
    :param dct_coeffs: DCT coefficients of shape [num_vertical_blocks * 8, num_horizontal_blocks * 8]
    :return: DCT coefficients reshaped to [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    """
    # From jpegio 2D to jpeglib 4D
    assert dct_coeffs.shape[0] % 8 == 0
    assert dct_coeffs.shape[1] % 8 == 0

    num_vertical_blocks = dct_coeffs.shape[0] // 8
    num_horizontal_blocks = dct_coeffs.shape[1] // 8

    # Reshape from [num_vertical_blocks * 8, num_horizontal_blocks * 8] to [num_vertical_blocks, 8, num_horizontal_blocks, 8]
    dct_coeffs = dct_coeffs.reshape((num_vertical_blocks, 8, num_horizontal_blocks, 8))

    # Reorer to [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    dct_coeffs = dct_coeffs.transpose((0, 2, 1, 3))

    return dct_coeffs
