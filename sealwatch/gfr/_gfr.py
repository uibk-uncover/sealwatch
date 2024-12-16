"""
Implementation of the GFR features as described in

Xiaofeng Song, Fenlin Liu, Chunfang Yang, Xiangyang Luo, and Yi Zhang
"Steganalysis of Adaptive JPEG Steganography Using 2D Gabor Filters"
ACM Workshop on Information Hiding and Multimedia Security 2015
https://doi.org/10.1145/2756601.2756608

Author: Benedikt Lorch
Affiliation: University of Innsbruck

This implementation builds on the Matlab implementation provided by the DDE lab. Please find the license of their implementation below.
-------------------------------------------------------------------------
Copyright (c) 2015 DDE Lab, Binghamton University, NY. All Rights Reserved.
Permission to use, copy, modify, and distribute this software for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that this copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from DDE Lab. DDE Lab does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall Binghamton University or DDE Lab be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software. DDE Lab disclaims any warranties, and has no obligations to provide maintenance, support, updates, enhancements or modifications.
-------------------------------------------------------------------------
"""  # noqa: E501


from collections import OrderedDict
import enum
import numpy as np
import os
from pathlib import Path
import scipy.signal
from typing import Union

from .. import dctr
from .. import tools

# from scipy.signal import fftconvolve
# from sealwatch.utils.jpeg import decompress_luminance_from_file
# from sealwatch.features.dctr.dctr import get_symmetric_histogram_coordinates
# from sealwatch.utils.matlab import matlab_round


class Implementation(enum.Enum):
    """GFR implementation to choose from."""

    GFR_ORIGINAL = enum.auto()
    """Original GFR implementation by DDE."""
    GFR_FIX = enum.auto()
    """GFR implementation with fixes."""


def compute_gabor_kernel(
    sigma: float,
    theta: float,
    phi: float,
    gamma: float,
    *,
    implementation: Implementation = Implementation.GFR_ORIGINAL,
    lambda_: float = None,
) -> np.ndarray:
    """Compute Gabor filter kernel according to Eq. 2.
    Gabor filters are a set of differently oriented sinusoidal patterns modulated by a Gaussian kernel.

    :param sigma: scale parameter.
        A small sigma means high spatial resolution so that the filtered coefficients reflect local properties in fine scale.
        A large sigma means low spatial resolution so that the coefficients reflect local properties in coarse scale.
    :type sigma: float
    :param theta: orientation angle of the 2D Gabor filter in radians
    :type theta: float
    :param phi: phase offset of the cosine factor.
        phi = 0 pi corresponds to symmetric "centre-on" functions.
        phi = -pi / 2, pi / 2 corresponds to anti-symmetric functions
    :type phi: float
    :param gamma: spatial aspect ratio and specifies the ellipticity of Gaussian factor
    :type gamma: float
    :param implementation:
    :type implementation: `Implementation`
    :param lambda_: wavelength of the cosine factor. If not given, lambda is set to sigma / 0.56.
    :type lambda_: float
    :return: 2D Gabor filter kernel of size 8x8
    :rtype:

    :Example:

    >>> # TODO
    """

    if lambda_ is None:
        lambda_ = sigma / 0.56

    gamma_squared = gamma ** 2
    s = 1 / (2 * sigma ** 2)
    f = 2 * np.pi / lambda_

    # Sampling points for Gabor function
    x, y = np.meshgrid(np.linspace(-7/2, 7/2, 8), -np.linspace(-7/2, 7/2, 8))

    xp = +x * np.cos(theta) + y * np.sin(theta)
    yp = -x * np.sin(theta) + y * np.cos(theta)
    kernel = np.exp(-s * (xp * xp + gamma_squared * yp * yp)) * np.cos(f * xp + phi)

    # Normalization
    # The original paper states that the kernel is zero-meaned by subtracting the kernel mean.
    # We copy the normalization from the DDE lab, although it seems to miss parentheses.
    # TODO: Missing parentheses?
    if implementation is Implementation.GFR_ORIGINAL:
        kernel = kernel - np.sum(kernel) / np.sum(np.abs(kernel)) * np.abs(kernel)
    elif implementation is Implementation.GFR_FIX:
        kernel = (kernel - np.sum(kernel)) / np.sum(np.abs(kernel)) * np.abs(kernel)
    else:
        raise NotImplementedError(f'unknown implementation {implementation}')
    # In a related publication, the DDE lab normalizes their Gabor kernels to zero mean by subtracting the kernel mean from all its elements.
    # kernel = kernel - np.mean(kernel)
    # http://dde.binghamton.edu/tomasD/pdf/WIFS2014_Selection-Channel-Aware_Rich_Model_for_Steganalysis_of_Digital_Images.pdf

    return kernel


def extract(
    img: np.ndarray,
    *,
    num_rotations: int = 32,
    quantization_steps: int = 75,
    T: int = 4,
    implementation: Implementation = Implementation.GFR_ORIGINAL,
) -> np.ndarray:
    """
    Extract the Gabor filter residual features from a given image.

    :param img: grayscale image with values in range [0, 255]
    :type img:
    :param num_rotations: number of rotations for Gabor kernel
    :type num_rotations: int
    :param quantization_steps: quantization step for each of the four scales
    :type quantization_steps: int
    :param T: the highest histogram bin value after quantization. The histogram contains T + 1 bins corresponding to the values [0, ..., T]. Quantized values exceeding T will be clamped to T.
    :type T: int
    :param implementation:
    :type implementation: `Implementation`
    :return: extracted Gabor features as 5D ndarray. The five dimensions denote:
        # Dimension 0: Phase shifts
        # Dimension 1: Scales
        # Dimension 2: Rotations/Orientations
        # Dimension 3: Number of histograms
        # Dimension 4: Co-occurrences

        Flatten the 5D array to obtain a 1D feature descriptor.

        Will be changed in the future to OrderedDict to match the common interface.
    :rtype:

    :Example:

    >>> # TODO
    """
    assert len(quantization_steps) == 4, "Expected four quantization steps, one for each scale"

    # Set up number of orientations from [0, 180) degree.
    # The number of rotations is denoted as S in the paper.
    assert num_rotations % 2 == 0, "Expected number of rotations to be an even number"
    rotations = np.arange(num_rotations) * np.pi / num_rotations

    # Set up number of scales
    # Standard deviations
    sigmas = [0.5, 0.75, 1, 1.25]
    # The number of scales is denoted as L in the paper.
    num_scales = len(sigmas)

    # The [original paper](https://ieeexplore.ieee.org/document/1042386) actually uses -pi/2
    phase_shifts = [0, np.pi / 2]
    num_phase_shifts = len(phase_shifts)

    aspect_ratio = 0.5

    # Compute DCTR locations to be merged
    merged_coordinates = dctr._dctr.get_symmetric_histogram_coordinates()

    # The bins correspond to values [0, ..., T]
    # np.histogram requires bin edges
    # Bin 0: [-0.5, 0.5]
    # Bin 1: [ 0.5, 1.5]
    # ...
    # Bin T: [T-0.5, T+0.5]
    bin_edges = np.arange(T + 2) - 0.5

    # In total, the number of 2D Gabor filters is 2 * L * S = 2 phase shifts * 32 rotations * 4 scales = 256.
    # For each Gabor filter, we have 25 histograms, each of which contains T + 1 values.
    # That results in 32000 feature dimensions before symmetrization.
    intermediate_features = np.zeros((num_phase_shifts, num_scales, num_rotations, len(merged_coordinates), T + 1))

    # Iterate over phase shifts
    for phase_idx, phase_shift in enumerate(phase_shifts):

        # Iterate over scales
        for scale_idx, sigma in enumerate(sigmas):

            # Iterate over orientations
            for rotation_idx, rotation in enumerate(rotations):

                # Compute Gabor kernel
                kernel = compute_gabor_kernel(
                    sigma,
                    rotation,
                    phase_shift,
                    aspect_ratio,
                    implementation=implementation,
                )

                # Convolve image with kernel
                R = scipy.signal.fftconvolve(img, kernel, mode='valid')

                # Quantization
                R = np.abs(tools.matlab.round(R / quantization_steps[scale_idx]))

                # Truncation
                R[R > T] = T

                # Feature extraction and merging, similar to DCTR features
                # Iterate over the output histograms
                # merged_coordinates contains a list of lists, where the inner list contains between 0 and 4 coordinate tuples
                for hist_idx in np.arange(len(merged_coordinates)):
                    sub_features = np.zeros(T + 1, dtype=int)

                    for (row_shift, col_shift) in merged_coordinates[hist_idx]:
                        # Get quantized and truncated result at given phase shift
                        R_sub = R[row_shift::8, col_shift::8]

                        # Compute histogram and add up
                        sub_features += np.histogram(R_sub, bin_edges)[0]

                    # Assign to output features array
                    intermediate_features[phase_idx, scale_idx, rotation_idx, hist_idx, :] = sub_features / np.sum(sub_features)

    # Prepare for merging histograms with identical scale and symmetrical direction
    # Examples: theta = pi / 8 and 7 pi / 8, as well as theta = 2 pi / 8 and 6 pi / 8
    # The justification is that images have similar statistical characteristics in symmetrical orientations.
    # To my understanding, the assumption is that images are invariant to flipping.

    # At this point, we have 2 phase shifts * 32 rotations * 4 scales * 25 histograms * (T=4 + 1) values.
    # Now we merge S rotations into (S/2 + 1) values.

    # Dimensions of final feature descriptor
    # Dimension 0: Phase shifts
    # Dimension 1: Scales
    # Dimension 2: Rotations/Orientations
    # Dimension 3: Number of histograms
    # Dimension 4: Co-occurrences
    final_shape = (
        num_phase_shifts,
        num_scales,
        num_rotations // 2 + 1,
        len(merged_coordinates),
        T + 1)

    # Allocate space
    features = np.zeros(final_shape, dtype=float)

    # Copy features for orientation 0
    features[:, :, 0, :, :] = intermediate_features[:, :, 0, :, :]

    # Copy features for orientation pi / 2
    features[:, :, -1, :, :] = intermediate_features[:, :, num_rotations // 2, :, :]

    # Merge orientations i / pi and (pi - i) / pi
    rotation_indices = np.arange(1, num_rotations // 2)
    features[:, :, rotation_indices, :, :] = (intermediate_features[:, :, rotation_indices, :, :] + intermediate_features[:, :, -rotation_indices, :, :]) / 2

    return features


def extract_from_file(
    path: Union[Path, str],
    num_rotations: int = 32,
    qf: int = None,
    quantization_steps: int = None,
    T: int = 4,
    implementation: Implementation = Implementation.GFR_ORIGINAL,
) -> OrderedDict:
    """
    Extract the Gabor filter residual features from a given JPEG image file.

    :param path: filepath to JPEG image
    :param num_rotations: number of rotations for Gabor kernel
    :param qf: JPEG quality factor; used to select the quantization steps
    :param quantization_steps: list of four quantization steps, one for each scale
    :param T: truncation threshold
    :return: extracted Gabor features as 5D ndarray. The five dimensions denote:
        # Dimension 0: Phase shifts
        # Dimension 1: Scales
        # Dimension 2: Rotations/Orientations
        # Dimension 3: Number of histograms
        # Dimension 4: Co-occurrences

        Flatten the 5D array to obtain a 1D feature descriptor.
    :rtype:
    """
    if quantization_steps is not None:
        if not len(quantization_steps) == 4:
            raise ValueError("Expected four quantization steps, one per scale")

    else:
        if qf is None:
            raise ValueError("Expected either quantization steps or quality factor to be given")

        # Select quantization steps based on quality factor
        if qf == 75:
            quantization_steps = [2, 4, 6, 8]
        elif qf == 95:
            quantization_steps = [0.1, 1, 1.5, 2]
        else:
            raise ValueError("Supported quality factors are 75 and 95")

    # Decompress image without rounding
    img = tools.jpeg.decompress_from_file(path)
    return extract(
        img,
        num_rotations=num_rotations,
        quantization_steps=quantization_steps,
        T=T,
        implementation=implementation,
    )


def generate_subspace_description(idx, num_rotations=32, T=4):
    # Dimension 0: Phase shifts (2 values)
    # Dimension 1: Scales (4 values)
    # Dimension 2: Rotations/Orientations (32/2 + 1 values)
    # Dimension 3: Number of histograms (25 values)
    # Dimension 4: Co-occurrences (5 values)
    shape = (2, 4, num_rotations, 25, 5)

    phase_shift_idx, scale_idx, rotation_idx, hist_idx, bin_idx = np.unravel_index(idx, shape=shape)

    phase_shifts = [0, np.pi / 2]
    scales = [0.5, 0.75, 1, 1.25]
    rotations = np.arange(num_rotations) * np.pi / num_rotations
    co_occurrences = np.arange(T + 1)

    return {
        "phase_shift": phase_shifts[phase_shift_idx],
        "scale": scales[scale_idx],
        "rotation": rotations[rotation_idx] / np.pi,
        "histogram": hist_idx,
        "co_occurrence": co_occurrences[bin_idx],
    }
