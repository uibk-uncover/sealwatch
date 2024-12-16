"""
Implementation of the cc-JRM features as described in

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
from typing import Union

from . import jrm


def extract(
    y1: np.ndarray,
    *,
    qt: np.ndarray = None,
) -> np.ndarray:
    """Extracts calibrated JPEG rich models (JRM) for the given DCT coefficients.

    :param y1: DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: cc-JRM features as ordered dict, where the keys are the names of the submodels. All submodels together have dimensionality 22510
    :rtype: collections.OrderedDict
    """
    return jrm.extract(y1=y1, calibrated=True, qt=qt)


def extract_from_file(
    path: Union[Path, str],
) -> OrderedDict:
    """
    Compute the clibrated JPEG rich models (cc-JRM) feature descriptor from the given image's luminance channel.

    The mode-specific submodels give the rich model a fine "granularity" at the price of utilizing only a small portion of the DCT plane.
    To cover a larger range of DCT coefficients, the mode-specific submodels are complemented by co-occurrence matrices integrated over all DCT modes.

    J. Kodovsky, J. Fridrich, Steganalysis of JPEG Images Using Rich Models, Proc. SPIE, Electronic Imaging, Media Watermarking, Security, and Forensics XIV, San Francisco, CA, January 23–25, 2012.
    http://dde.binghamton.edu/kodovsky/pdf/SPIE2012_Kodovsky_Steganalysis_of_JPEG_Images_Using_Rich_Models_paper.pdf

    :param path: path to JPEG image
    :type path:
    :return: cc-JRM features as ordered dict, where the keys are the names of the submodels. All submodels together have dimensionality 22510
    :rtype: collections.OrderedDict
    """
    return jrm.extract_from_file(path=path, calibrated=True)
