"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import jpeglib
import numpy as np
from tempfile import NamedTemporaryFile
from typing import Tuple


# def decompress_crop_recompress(input_filepath, output_filepath):
#     """
#     Decompress the given image, crop 4 pixels from all sides, then recompress using the same quantization table
#     :param input_filepath: path to JPEG image
#     :param output_filepath: where to store the resulting JPEG images
#     """

#     # Decompress into spatial domain
#     im_spatial = jpeglib.read_spatial(input_filepath)
#     im_dct = jpeglib.read_dct(input_filepath)

#     img = im_spatial.spatial

#     # Crop 4 pixels from all sides
#     img = img[4:-4, 4:-4]

#     # Recompress
#     im = jpeglib.from_spatial(img.copy())
#     im.write_spatial(output_filepath, qt=im_dct.qt)


def cartesian(
    y1: np.ndarray,
    qt: np.ndarray,
    *,
    crop: Tuple[int] = (4, -4),
) -> np.ndarray:
    """Performs cartesian calibration of given DCT.

    :param y1: Stego DCT coefficients.
    :type y1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: Quantization table.
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param crop: Crop to apply in the decompressed domain. [4:-4, 4:-4] by default.
    :type qt: tuple of int
    :return: Calibrated DCT coefficients.
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> y2 = sw.utils.calibration.cartesian(y1, jpeg1.qt[0])
    """
    #
    with NamedTemporaryFile(suffix='jpeg') as tmp:

        # decompress
        jpeglib.from_dct(
            Y=y1,
            qt=qt,
        ).write_dct(tmp.name)

        # crop 4x4
        x1 = jpeglib.read_spatial(tmp.name).spatial
        x2 = x1[crop[0]:crop[1], crop[0]:crop[1]]

        # compress
        jpeglib.from_spatial(
            x2,
        ).write_spatial(tmp.name, qt=qt)

        # read DCT
        return jpeglib.read_dct(tmp.name).Y
