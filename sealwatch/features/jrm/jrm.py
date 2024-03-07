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

import tempfile
import jpeglib
import numpy as np
from sealwatch.utils.jpeg import jpeglib_to_jpegio
from sealwatch.features.jrm.absolute_value_features import compute_absolute_value_features
from sealwatch.features.jrm.difference_features import compute_difference_features
from sealwatch.features.jrm.integral_features import compute_integral_features
from sealwatch.utils.calibration import decompress_crop_recompress
from sealwatch.utils.dict import append_features
from collections import OrderedDict


def extract_cc_jrm_features_from_file(img_filepath):
    """
    Compute JPEG rich model (JRM) features including reference features from a cartesian-calibrated variant of the input image.
    Uses only the luminance channel

    :param img_filepath: path to JPEG image
    :return: ccJRM features as ordered dict, where the keys are the names of the submodels. All submodels together have dimensionality 22510.
    """

    # Compute JRM features from given image
    jrm_features = extract_jrm_features_from_file(img_filepath)

    # Cartesian calibration
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        decompress_crop_recompress(img_filepath, f.name)

        # Extract JRM features from reference image
        cc_features = extract_jrm_features_from_file(f.name)

    # Copy features to the JRM features dict, but append the suffix "_ref" to the submodel names
    for name, submodel in cc_features.items():
        new_name = f"{name}_ref"
        jrm_features[new_name] = submodel

    return jrm_features


def extract_jrm_features_from_file(img_filepath):
    """
    Compute the JPEG rich models (JRM) feature descriptor from the given image's luminance channel.

    The mode-specific submodels give the rich model a fine "granularity" at the price of utilizing only a small portion of the DCT plane.
    To cover a larger range of DCT coefficients, the mode-specific submodels are complemented by co-occurrence matrices integrated over all DCT modes.

    J. Kodovsky, J. Fridrich, Steganalysis of JPEG Images Using Rich Models, Proc. SPIE, Electronic Imaging, Media Watermarking, Security, and Forensics XIV, San Francisco, CA, January 23–25, 2012.
    http://dde.binghamton.edu/kodovsky/pdf/SPIE2012_Kodovsky_Steganalysis_of_JPEG_Images_Using_Rich_Models_paper.pdf

    :param img_filepath: path to JPEG image
    :return: JRM features as ordered dict, where the keys are the names of the submodels. All submodels together have dimensionality 11255
    """

    # Read the luminance channel
    im = jpeglib.read_dct(img_filepath)
    luminance_dct_coeffs = im.Y

    return extract_jrm_features_from_img(dct_coeffs=luminance_dct_coeffs)


def extract_jrm_features_from_img(dct_coeffs):
    """
    Compute the JPEG rich models (JRM) feature descriptor from the given DCT coefficients

    The mode-specific submodels give the rich model a fine "granularity" at the price of utilizing only a small portion of the DCT plane.
    To cover a larger range of DCT coefficients, the mode-specific submodels are complemented by co-occurrence matrices integrated over all DCT modes.

    J. Kodovsky, J. Fridrich, Steganalysis of JPEG Images Using Rich Models, Proc. SPIE, Electronic Imaging, Media Watermarking, Security, and Forensics XIV, San Francisco, CA, January 23–25, 2012.
    http://dde.binghamton.edu/kodovsky/pdf/SPIE2012_Kodovsky_Steganalysis_of_JPEG_Images_Using_Rich_Models_paper.pdf
    Presentation slides with an illustration of the submodels: http://dde.binghamton.edu/kodovsky/pdf/SPIE2012_Kodovsky_Steganalysis_of_JPEG_Images_Using_Rich_Models_slides.pdf

    :param dct_coeffs: DCT coefficient array of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :return: JRM features as ordered dict, where the keys are the names of the submodels. All submodels together have dimensionality 11255
    """

    # Save features as ordered dict
    features = OrderedDict()

    abs_X = np.abs(dct_coeffs)

    # Work with the 2D format
    abs_X = jpeglib_to_jpegio(abs_X)
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
    append_features(features, features_diff1, prefix="intra_block_hv")

    # DCT-mode specific co-occurrences of differences of absolute values (diagonal)
    # G^{south-east arrow}: Intra-block differences (diagonal), 2041 features
    features_diff2 = compute_difference_features(abs_X_Dd, abs_X_Dd, T=2)

    # Copy local features to global features buffer
    append_features(features, features_diff2, prefix="intra_block_diag")

    # DCT-mode specific co-occurrences of differences of absolute values (inter-block horizontal/vertical)
    # G^{double arrow east}: Inter-block differences (horizontal/vertical), 2041 features
    features_diff3 = compute_difference_features(abs_X_Dih, abs_X_Div, T=2)

    # Copy local features to global features buffer
    append_features(features, features_diff3, prefix="inter_block_hv")

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
