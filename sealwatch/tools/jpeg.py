"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import jpeglib
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Union

from .dct import block_idct2, jpeglib_to_jpegio


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


def qf_to_qt(
    qf: int,
    *,
    version: str = '6b',
) -> np.ndarray:
    """Compress a dummy color image with the given quality factor and load its quantization table

    :param qf: JPEG quality factor
    :type qf: int
    :param version: libjpeg version to be passed to jpeglib
    :type version: str
    :return: quantization tables used by the selected libjpeg version
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    # dummy image
    x = np.random.randint(low=0, high=256, dtype=np.uint8, size=(64, 64, 3))
    im = jpeglib.from_spatial(x)

    # compress
    with NamedTemporaryFile(suffix=".jpg") as f:
        with jpeglib.version(version):
            im.write_spatial(f.name, qt=qf)

        # extract QT
        return jpeglib.read_dct(f.name).qt


def create_qt_to_qf_mapping(
    version: str = '6b',
    grayscale: bool = False,
) -> Dict[str, int]:
    """
    Iterate over all JPEG quality factors and store the quantization tables in a dictionary.
    The keys are the 3D quantization tables converted to a string.
    For simplicity, also for grayscale images the keys are strings of 3D quantization tables with shape [1, 8, 8].

    :param libjpeg_version: libjpeg version to be passed to jpeglib
    :param grayscale: if True, the keys are the luminance QTs only. If false (default), concatenate both luminance and chrominance QTs as keys.
    :return: dict where the keys are the quantization tables encoded as string, and the values are the corresponding quality factors.
        The keys are strings from 3D matrices of shape [1, 8, 8] for grayscale images or [2, 8, 8] for color images.
    """

    mapping = {}
    for quality in range(0, 101):
        qts = qf_to_qt(quality, version=version)

        if grayscale:
            # Select only the luminance QT
            qts = qts[:1]

        key = str(qts)
        mapping[key] = quality

    return mapping


def identify_qf(
    path: Union[Path, str],
    qt_to_qf_map: Dict[str, int] = None,
) -> int:
    """
    Identify the JPEG quality factor from a given JPEG file by comparing it to a set of known quantization tables
    :param filepath: path to JPEG file
    :param qt_to_qf_map: dict where the keys are the quantization tables encoded as string, and the values are the corresponding quality factors.
    :return: JPEG quality factor, or None if not present in the given map
    """
    qts = jpeglib.read_dct(path).qt
    key = str(qts)
    is_grayscale = len(qts) == 1

    if qt_to_qf_map is None:
        qt_to_qf_map = create_qt_to_qf_mapping(grayscale=is_grayscale)

    if key not in qt_to_qf_map:
        print(f"Could not find quality for image \"{path}\"")
        return None

    return qt_to_qf_map.get(key, None)
