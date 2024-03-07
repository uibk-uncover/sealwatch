import numpy as np
import jpeglib
from sealwatch.utils.dct import block_idct2


def decompress_channel(dct_coeffs, quantization_table=None, dct_mat=None):
    """
    Decompress a single channel
    :param dct_coeffs: quantized DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :param quantization_table: quantization table of shape [8, 8]
    :param dct_mat: pre-computed DCT matrix. If None, the DCT matrix will be computed on the fly.
    :return: decompressed channel of shape [num_vertical_blocks * 8, num_horizontal_blocks * 8]. The values should be in the range [0, 255], although some values may exceed this range.
    """

    # Validate quantization table
    if quantization_table is not None:
        assert (
                len(quantization_table.shape) == 2
                and quantization_table.shape[0] == 8
                and quantization_table.shape[1] == 8), "Expected quantization table of shape [8, 8]"

    num_vertical_blocks, num_horizontal_blocks = dct_coeffs.shape[:2]

    # If quantization table is given, unquantize the DCT coefficients
    if quantization_table is not None:
        dct_coeffs = dct_coeffs * quantization_table[None, None, :, :]

    # Inverse DCT
    spatial_blocks = block_idct2(dct_coeffs, dct_mat=dct_mat)

    # Reorder blocks to obtain image
    spatial = np.transpose(spatial_blocks, axes=[0, 2, 1, 3]).reshape(num_vertical_blocks * 8, num_horizontal_blocks * 8)

    # Level shift
    spatial += 128

    return spatial


def decompress_luminance_from_file(filepath, dct_mat=None):
    """
    Decompress only the luminance channel from the given JPEG image
    :param filepath: path to JPEG image
    :param dct_mat: pre-computed DCT matrix. If None, will be computed on the fly.
    :return: decompressed luminance channel, ndarray of shape [height, width]
    """
    im_jpeglib = jpeglib.read_dct(filepath)

    img_height = im_jpeglib.height
    img_width = im_jpeglib.width

    quantization_table = im_jpeglib.get_component_qt(0)

    luminance = decompress_channel(dct_coeffs=im_jpeglib.Y, quantization_table=quantization_table, dct_mat=dct_mat)

    # Crop pixels that exceed the image boundary
    luminance_height, luminance_width = luminance.shape
    if luminance_height - img_height > 0:
        assert luminance_height - img_height < 8, "Expected less than 8 pixels"
        luminance = luminance[:img_height]

    if luminance_width - img_width > 0:
        assert luminance_width - img_width < 8, "Expected less than 8 pixels"
        luminance = luminance[:, :img_width]

    return luminance


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
