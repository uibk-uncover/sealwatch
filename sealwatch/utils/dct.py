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
