"""
Implementation of the JRM features as described in

Jan Kodovský and Jessica Fridrich
"Steganalysis of JPEG images using rich models"
IS&T/SPIE Electronic Imaging, 2012
https://doi.org/10.1117/12.907495

Author: Benedikt Lorch
Affiliation: University of Innsbruck

This implementation builds on the original Matlab implementation provided by the paper authors. Please find the license of the original implementation below.
-------------------------------------------------------------------------
Copyright (c) 2011 DDE Lab, Binghamton University, NY. All Rights Reserved.
Permission to use, copy, modify, and distribute this software for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that this copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from DDE Lab. DDE Lab does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall Binghamton University or DDE Lab be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software. DDE Lab disclaims any warranties, and has no obligations to provide maintenance, support, updates, enhancements or modifications.
-------------------------------------------------------------------------
"""  # noqa: E501

from collections import OrderedDict
import jpeglib
import numpy as np
from pathlib import Path
from typing import Union, Dict

from .absolute_value_features import compute_absolute_value_features
from .difference_features import compute_difference_features
from .integral_features import compute_integral_features
from .. import tools


def _extract_features(
    y: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extracts JPEG rich models (JRM) for the given DCT coefficients.

    The mode-specific submodels give the rich model a fine "granularity" at the price of utilizing only a small portion of the DCT plane.
    To cover a larger range of DCT coefficients, the mode-specific submodels are complemented by co-occurrence matrices integrated over all DCT modes.

    J. Kodovsky, J. Fridrich, Steganalysis of JPEG Images Using Rich Models, Proc. SPIE, Electronic Imaging, Media Watermarking, Security, and Forensics XIV, San Francisco, CA, January 23–25, 2012.
    http://dde.binghamton.edu/kodovsky/pdf/SPIE2012_Kodovsky_Steganalysis_of_JPEG_Images_Using_Rich_Models_paper.pdf
    Presentation slides with an illustration of the submodels: http://dde.binghamton.edu/kodovsky/pdf/SPIE2012_Kodovsky_Steganalysis_of_JPEG_Images_Using_Rich_Models_slides.pdf

    :param y: DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: JRM features as ordered dict, where the keys are the names of the submodels. All submodels together have dimensionality 11255.
    :rtype: collections.OrderedDict
    """

    # Save features as ordered dict
    features = OrderedDict()

    abs_X = np.abs(y)

    # Work with the 2D format
    abs_X = tools.jpeglib_to_jpegio(abs_X)
    height, width = abs_X.shape

    # DCT-mode specific co-occurrences of absolute values
    # G^x: 2512 features
    features_abs_values = compute_absolute_value_features(abs_X, T=3)

    # Copy features
    features.update(features_abs_values)

    # DCT-mode specific co-occurrences of differences of absolute values (horizontal/vertical)
    # Horizontal direction
    abs_X_Dh = abs_X[:, :width - 8] - abs_X[:, 1:width - 7]

    # Vertical direction
    abs_X_Dv = abs_X[:height - 8, :] - abs_X[1:height - 7, :]

    # Major diagonal
    abs_X_Dd = abs_X[:height - 8, :width - 8] - abs_X[1:height - 7, 1:width - 7]

    # Inter-block horizontal
    abs_X_Dih = abs_X[:, :width - 8] - abs_X[:, 8:]

    # Inter-block vertical
    abs_X_Div = abs_X[:height - 8, :] - abs_X[8:, :]

    # G^{east arrow}: Intra-block differences (horizontal/vertical), 2041 features
    features_diff1 = compute_difference_features(abs_X_Dh, abs_X_Dv, T=2)

    # Copy local features to global features buffer
    tools.features.append(features, features_diff1, prefix="intra_block_hv")

    # DCT-mode specific co-occurrences of differences of absolute values (diagonal)
    # G^{south-east arrow}: Intra-block differences (diagonal), 2041 features
    features_diff2 = compute_difference_features(abs_X_Dd, abs_X_Dd, T=2)

    # Copy local features to global features buffer
    tools.features.append(features, features_diff2, prefix="intra_block_diag")

    # DCT-mode specific co-occurrences of differences of absolute values (inter-block horizontal/vertical)
    # G^{double arrow east}: Inter-block differences (horizontal/vertical), 2041 features
    features_diff3 = compute_difference_features(abs_X_Dih, abs_X_Div, T=2)

    # Copy local features to global features buffer
    tools.features.append(features, features_diff3, prefix="inter_block_hv")

    # Integral features
    # I: 2620 features
    integral_features = compute_integral_features(
        abs_X=abs_X,
        abs_X_Dh=abs_X_Dh,
        abs_X_Dv=abs_X_Dv,
        abs_X_Dd=abs_X_Dd,
        abs_X_Dih=abs_X_Dih,
        abs_X_Div=abs_X_Div,
        T=5)

    # Copy local features over into global buffer
    features.update(integral_features)

    return features


def extract(
    y1: np.ndarray,
    *,
    calibrated: bool = False,
    qt: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Extracts JPEG rich models (JRM) for the given DCT coefficients.

    :param y1: DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param calibrated: Choose JRM or cc-JRM.
    :type calibrated: bool
    :param qt: quantization table
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: JRM features as ordered dict, where the keys are the names of the submodels. All submodels together have dimensionality 11255
    :rtype: collections.OrderedDict
    """

    # extract features
    f = _extract_features(y=y1)

    # extract calibrated features
    if calibrated:
        y2 = tools.calibration.cartesian(
            y1=y1,
            qt=qt,
        )
        f_y2 = _extract_features(y=y2)
        for name, submodel in f_y2.items():
            f[f'{name}_ref'] = submodel

    #
    return f


def extract_from_file(
    path: Union[Path, str],
    calibrated: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute the JPEG rich models (JRM) feature descriptor from the given image's luminance channel.

    The mode-specific submodels give the rich model a fine "granularity" at the price of utilizing only a small portion of the DCT plane.
    To cover a larger range of DCT coefficients, the mode-specific submodels are complemented by co-occurrence matrices integrated over all DCT modes.

    J. Kodovsky, J. Fridrich, Steganalysis of JPEG Images Using Rich Models, SPIE, Electronic Imaging, Media Watermarking, Security, and Forensics, 2012.
    http://dde.binghamton.edu/kodovsky/pdf/SPIE2012_Kodovsky_Steganalysis_of_JPEG_Images_Using_Rich_Models_paper.pdf

    :param path: path to JPEG image
    :type path:
    :param calibrated: Choose JRM or cc-JRM.
    :type calibrated: bool
    :return: JRM features as ordered dict, where the keys are the names of the submodels. All submodels together have dimensionality 11255
    :rtype: collections.OrderedDict
    """
    # read file
    jpeg1 = jpeglib.read_dct(path)
    y1 = jpeg1.Y
    qt = jpeg1.qt[jpeg1.quant_tbl_no[0]]

    # extract features
    return extract(y1=y1, qt=qt, calibrated=calibrated)
