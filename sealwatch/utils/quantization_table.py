import jpeglib
import tempfile
import numpy as np


def qf_to_qt(qf, libjpeg_version="6b"):
    """
    Compress a dummy color image with the given quality factor and load its quantization table
    :param qf: JPEG quality factor
    :param libjpeg_version: libjpeg version to be passed to jpeglib
    :return: quantization tables used by the selected libjpeg version
    """
    dummy_img = np.random.randint(low=0, high=256, dtype=np.uint8, size=(64, 64, 3))

    im = jpeglib.from_spatial(dummy_img)

    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        with jpeglib.version(libjpeg_version):
            im.write_spatial(f.name, qt=qf)

        return jpeglib.read_dct(f.name).qt


def create_qt_to_qf_mapping(libjpeg_version="6b", grayscale=False):
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
        qts = qf_to_qt(quality, libjpeg_version=libjpeg_version)

        if grayscale:
            # Select only the luminance QT
            qts = qts[:1]

        key = str(qts)
        mapping[key] = quality

    return mapping


def identify_qf(filepath, qt_to_qf_map=None):
    """
    Identify the JPEG quality factor from a given JPEG file by comparing it to a set of known quantization tables
    :param filepath: path to JPEG file
    :param qt_to_qf_map: dict where the keys are the quantization tables encoded as string, and the values are the corresponding quality factors.
    :return: JPEG quality factor, or None if not present in the given map
    """
    qts = jpeglib.read_dct(filepath).qt
    key = str(qts)
    is_grayscale = len(qts) == 1

    if qt_to_qf_map is None:
        qt_to_qf_map = create_qt_to_qf_mapping(grayscale=is_grayscale)

    if key not in qt_to_qf_map:
        print(f"Could not find quality for image \"{filepath}\"")
        return None

    return qt_to_qf_map.get(key, None)
