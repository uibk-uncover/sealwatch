"""

Authors: Benedikt Lorch
Affiliation: University of Innsbruck
"""

from collections import OrderedDict
import enum
import numpy as np
from pathlib import Path
from typing import Union, Any

from . import pharm_original
from . import pharm_revisited
from .. import tools


class Implementation(enum.Enum):
    """PHARM implementation to choose from."""

    PHARM_ORIGINAL = enum.auto()
    """Original PHARM implementation by DDE."""
    PHARM_REVISITED = enum.auto()
    """PHARM implementation with fixes."""


PHARM_ORIGINAL = Implementation.PHARM_ORIGINAL
PHARM_REVISITED = Implementation.PHARM_REVISITED


def get_extractor_factory(
    implementation: Implementation = Implementation.PHARM_REVISITED,
) -> Any:  # Extractor superclass
    #
    if implementation is Implementation.PHARM_REVISITED:
        construct_extractor = pharm_revisited.RevisitedExtractor
    elif implementation is Implementation.PHARM_ORIGINAL:
        construct_extractor = pharm_original.OriginalExtractor
    else:
        raise NotImplementedError(f'unknown implementation {implementation}')
    return construct_extractor


def qf_to_quantization_step(
    qf: int,
) -> int:
    return (65 / 4) - (3 / 20) * qf


def select_quantization_step(
    path: Union[str, Path],
) -> int:
    qf = tools.jpeg.identify_qf(path)
    return qf_to_quantization_step(qf)


def extract(
    x1: np.ndarray,
    *,
    implementation: Implementation = Implementation.PHARM_REVISITED,
    q: int = 5,
    T: int = 2,
    num_projections: int = 100,
    maximum_projection_size: int = 8,
    first_order_residuals: int = True,
    second_order_residuals: int = True,
    third_order_residuals: int = True,
    symmetrize: int = True,
    normalize: int = False,
    seed: int = 1,
) -> OrderedDict:
    """Extracts the PHARM features from a given decompressed image.

    The PHARM features were introduced in
    V. Holub and J. Fridrich, Phase-Aware Projection Model for Steganalysis of JPEG Images.
    SPIE Electronic Imaging, Media Watermarking, Security, and Forensics XVII, vol. 9409, 2015.
    http://dde.binghamton.edu/vholub/pdf/SPIE15_Phase-Aware_Projection_Model_for_Steganalysis_of_JPEG_Images.pdf

    :param x1: decompressed JPEG image
        of shape [height, width]
    :type x1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param implementation: implementation of PHARM to use
    :type implementation: `Implementation`
    :param q: quantization step
    :type q: int
    :param T: truncation threshold
    :type T: int
    :param num_projections: number of random projection matrices. The original implementation defaults to 900, but we use 100 for speed reasons.
    :type num_projections: int
    :param maximum_projection_size: maximum spatial size of each projection matrix
    :type maximum_projection_size: int
    :param first_order_residuals: If True, include first order residuals. If False, skip first order residuals.
    :type first_order_residuals: bool
    :param second_order_residuals: If True, include second order residuals. If False, skip second order residuals.
    :type second_order_residuals: bool
    :param third_order_residuals: If True, include third order residuals. If False, skip third order residuals.
    :type third_order_residuals: bool
    :param symmetrize: If True, merge histograms with horizontally and vertically flipped versions of the image. If False, skip symmetrization.
    :type symmetrize: bool
    :param normalize: If True, normalize the histogram counts.
    :type normalize: bool
    :param seed: seed for random number generator for the projection matrices
    :type seed: int
    :return: features as ordered dictionary, where the keys are the submodel names and the values are the features of shape [num_projections, T].
        Note that the features are not normalized.
    :rtype: `OrderedDict`

    :Example:

    >>> # TODO
    """
    #
    construct_extractor = get_extractor_factory(implementation=implementation)
    extractor = construct_extractor(
        q=q,
        T=T,
        num_projections=num_projections,
        maximum_projection_size=maximum_projection_size,
        first_order_residuals=first_order_residuals,
        second_order_residuals=second_order_residuals,
        third_order_residuals=third_order_residuals,
        symmetrize=symmetrize,
        normalize=normalize,
        seed=seed,
    )
    return extractor.extract(x1)


def extract_from_file(
    path: Union[str, Path],
    *,
    implementation: Implementation = Implementation.PHARM_REVISITED,
    q: int = 5,
    T: int = 2,
    num_projections: int = 100,
    maximum_projection_size: int = 8,
    first_order_residuals: int = True,
    second_order_residuals: int = True,
    third_order_residuals: int = True,
    symmetrize: int = True,
    normalize: int = False,
    seed: int = 1,
) -> OrderedDict:
    """Extracts the PHARM features from a given JPEG image.

    The PHARM features were introduced in
    V. Holub and J. Fridrich, Phase-Aware Projection Model for Steganalysis of JPEG Images.
    SPIE Electronic Imaging, Media Watermarking, Security, and Forensics XVII, vol. 9409, 2015.
    http://dde.binghamton.edu/vholub/pdf/SPIE15_Phase-Aware_Projection_Model_for_Steganalysis_of_JPEG_Images.pdf

    :param path: path to JPEG image
    :type path: str or `Path`
    :param implementation: implementation of PHARM to use
    :type implementation: `Implementation`
    :param q: quantization step
    :type q: int
    :param T: truncation threshold
    :type T: int
    :param num_projections: number of random projection matrices. The original implementation defaults to 900, but we use 100 for speed reasons.
    :type num_projections: int
    :param maximum_projection_size: maximum spatial size of each projection matrix
    :type maximum_projection_size: int
    :param first_order_residuals: whether to include first order residuals
    :type first_order_residuals: bool
    :param second_order_residuals: whether to include second order residuals
    :type second_order_residuals: bool
    :param third_order_residuals: whether to include third order residuals
    :type third_order_residuals: bool
    :param symmetrize: whether to merge histograms with horizontally and vertically flipped image. If False, skip symmetrization.
    :type symmetrize: bool
    :param normalize: whether to normalize the histogram counts, by default False
    :type normalize: bool
    :param seed: seed for random number generator for the projection matrices, by default 1
    :type seed: int
    :return: features as ordered dictionary, where the keys are the submodel names and the values are the features of shape [num_projections, T].
        Note that the features are not normalized.
    :rtype: `OrderedDict`

    :Example:

    >>> # TODO
    """
    #
    construct_extractor = get_extractor_factory(implementation=implementation)
    extractor = construct_extractor(
        q=q,
        T=T,
        num_projections=num_projections,
        maximum_projection_size=maximum_projection_size,
        first_order_residuals=first_order_residuals,
        second_order_residuals=second_order_residuals,
        third_order_residuals=third_order_residuals,
        symmetrize=symmetrize,
        normalize=normalize,
        seed=seed,
    )
    return extractor.extract_from_file(path)
