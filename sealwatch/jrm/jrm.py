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

from .common import MODES, direction_to_coordinates, normalize, sign_symmetrize
from .cooccurrence import cooccurrence, stack_columns
from .integral import compute_integral_features
from .. import tools


# DCT-mode specific submodels, see Section 2.2 of the paper,
# as (name, suffix, direction, mode ids).
# C(x, y, dx, dy) denotes the co-occurrence between the DCT modes (x, y) and (x + dx, y + dy).
SUBMODELS = [
    # intra-block relationships
    ('h', '', 'hor', list(range(1, 21))),  # { C(x, y, 0, 1) | x + y <= 5 }, DC omitted
    ('d', '_diag', 'diag', [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]),  # { C(x, y, 1, 1) | x <= y; x + y <= 5 }
    ('d', '_semidiag', 'semidiag', [6, 7, 8, 9, 10, 12, 13, 14, 17]),  # { C(x, y, 1, -1) | x <= y; x + y <= 5 }
    ('oh', '', 'hor_skip', [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 16, 18]),  # { C(x, y, 0, 2) | x + y <= 4 }
    ('x', '', 'sym_8x8', [1, 2, 3, 4, 5, 8, 9, 10, 14]),  # { C(x, y, y - x, x - y) | x < y; x + y <= 5 }
    ('od', '_diag_skip', 'diag_skip', [1, 2, 3, 4, 7, 8, 9, 13]),  # { C(x, y, 2, 2) | x <= y; x + y <= 4 }
    ('od', '_semidiag_skip', 'semidiag_skip', [11, 12, 13, 14, 22, 16, 17, 23, 24]),  # { C(x, y, 2, -2) | x < y; x + y <= 5 }
    ('m', '', 'horse', list(range(6, 21))),  # { C(x, y, -1, 2) | 1 <= x; x + y <= 5 }, knight move
    # inter-block relationships
    ('ih', '', 'inter_hor', list(range(1, 21))),  # { C(x, y, 0, 8) | x + y <= 5 }
    ('id', '', 'inter_diag', [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]),  # { C(x, y, 8, 8) | x <= y; x + y <= 5 }
    ('is', '', 'inter_semidiag', [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]),  # { C(x, y, -8, 8) | x <= y; x + y <= 5 }
    ('ix', '', 'inter_symm', list(range(1, 21))),  # { C(x, y, y - x, x - y + 8) | x + y <= 5 }
]


def _mode_specific_features(
    X1: np.ndarray,
    X2: np.ndarray,
    T: int,
    absolute: bool,
    prefix: str,
) -> OrderedDict:
    """DCT-mode specific submodels.

    Co-occurrences are computed from X1, and in the transposed direction from X2, and averaged.

    :param X1: absolute values or their differences, as 2D DCT plane
    :param X2: same as X1, used in the transposed direction
    :param T: truncation threshold
    :param absolute: absolute values, otherwise differences, which are sign-symmetrized
    :param prefix: prefix of the submodel names
    :return: ordered dict with submodels
    """
    features = OrderedDict()
    for name, suffix, direction, mode_ids in SUBMODELS:
        for mode_id in mode_ids:
            a, b = MODES[mode_id]
            targets = direction_to_coordinates(direction, a, b)
            f1 = normalize(cooccurrence(stack_columns(X1, targets), T, absolute=absolute))
            f2 = normalize(cooccurrence(stack_columns(X2, targets[:, ::-1]), T, absolute=absolute))
            if absolute:
                f = (f1 + f2).flatten(order='F') / 2
            else:
                f = normalize(sign_symmetrize(f1 + f2))
            features[f'{prefix}{name}_T{T}{suffix}_mode_y_{a}_x_{b}'] = f
    return features


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
    # absolute values, in the 2D format
    abs_X = tools.jpeglib_to_jpegio(np.abs(y))
    height, width = abs_X.shape

    # differences of absolute values
    D = {
        'H': abs_X[:, :width - 8] - abs_X[:, 1:width - 7],  # horizontal
        'V': abs_X[:height - 8, :] - abs_X[1:height - 7, :],  # vertical
        'D': abs_X[:height - 8, :width - 8] - abs_X[1:height - 7, 1:width - 7],  # diagonal
        'IH': abs_X[:, :width - 8] - abs_X[:, 8:],  # inter-block horizontal
        'IV': abs_X[:height - 8, :] - abs_X[8:, :],  # inter-block vertical
    }

    features = OrderedDict()
    # DCT-mode specific co-occurrences of absolute values, 2512 features
    features.update(_mode_specific_features(abs_X, abs_X, T=3, absolute=True, prefix='A'))
    # DCT-mode specific co-occurrences of differences, 3 x 2041 features
    for prefix, X1, X2 in [
        ('intra_block_hv_D', D['H'], D['V']),
        ('intra_block_diag_D', D['D'], D['D']),
        ('inter_block_hv_D', D['IH'], D['IV']),
    ]:
        features.update(_mode_specific_features(X1, X2, T=2, absolute=False, prefix=prefix))
    # integral features, 2620 features
    features.update(compute_integral_features(abs_X, D, T=5))
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

    :Example:

    >>> jpeg1 = jpeglib.read_dct('seal1.jpg')
    >>> features = sw.jrm.extract(jpeg1.Y, qt=jpeg1.qt[0])
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

    :Example:

    >>> features = sw.jrm.extract_from_file('seal1.jpg')
    """
    # read file
    jpeg1 = jpeglib.read_dct(path)
    y1 = jpeg1.Y
    qt = jpeg1.qt[jpeg1.quant_tbl_no[0]]

    # extract features
    return extract(y1=y1, qt=qt, calibrated=calibrated)
