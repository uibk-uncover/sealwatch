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
from pathlib import Path
import scipy.fft
from typing import Sequence, Union
import warnings

from .. import dctr
from .. import tools


# Standard deviations of the Gaussian factor, one per scale.
# The number of scales is denoted as L in the paper.
SIGMAS = [0.5, 0.75, 1, 1.25]
# The [original paper](https://ieeexplore.ieee.org/document/1042386) actually uses -pi/2
PHASE_SHIFTS = [0, np.pi / 2]
ASPECT_RATIO = 0.5


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


def compute_gabor_kernels(
    num_rotations: int = 32,
    *,
    implementation: Implementation = Implementation.GFR_ORIGINAL,
) -> np.ndarray:
    """Compute the bank of Gabor kernels used by GFR.

    :param num_rotations: number of rotations in [0, pi), denoted S in the paper
    :type num_rotations: int
    :param implementation:
    :type implementation: `Implementation`
    :return: Gabor kernels of shape [phase shifts, scales, rotations, 8, 8]
    :rtype: np.ndarray
    """
    kernels = np.zeros((len(PHASE_SHIFTS), len(SIGMAS), num_rotations, 8, 8))
    rotations = np.arange(num_rotations) * np.pi / num_rotations
    for phase_idx, phase_shift in enumerate(PHASE_SHIFTS):
        for scale_idx, sigma in enumerate(SIGMAS):
            for rotation_idx, rotation in enumerate(rotations):
                kernels[phase_idx, scale_idx, rotation_idx] = compute_gabor_kernel(
                    sigma,
                    rotation,
                    phase_shift,
                    ASPECT_RATIO,
                    implementation=implementation,
                )
    return kernels


def _extract(
    x: np.ndarray,
    *,
    num_rotations: int = 32,
    quantization_steps: Sequence[float] = (2, 4, 6, 8),
    T: int = 4,
    implementation: Implementation = Implementation.GFR_ORIGINAL,
) -> OrderedDict:
    """Python implementation of GFR, see extract()."""
    kernels = compute_gabor_kernels(num_rotations, implementation=implementation)
    num_phase_shifts, num_scales = kernels.shape[:2]
    height, width = x.shape

    # Compute DCTR locations to be merged
    merged_coordinates = dctr._dctr.get_symmetric_histogram_coordinates()

    # Phase (row, col) in the 8x8 grid of each residual, as a linear index
    rows, cols = np.mgrid[:height - 7, :width - 7]
    phases = (rows % 8) * 8 + cols % 8

    # The convolution is computed in the frequency domain,
    # the spectrum of the image is shared by all kernels
    fft_shape = [scipy.fft.next_fast_len(d) for d in (height + 7, width + 7)]
    x_spectrum = scipy.fft.fft2(x, fft_shape)

    # In total, the number of 2D Gabor filters is 2 * L * S = 2 phase shifts * 32 rotations * 4 scales = 256.
    # For each Gabor filter, we have 25 histograms, each of which contains T + 1 values.
    # That results in 32000 feature dimensions before symmetrization.
    intermediate_features = np.zeros((num_phase_shifts, num_scales, num_rotations, len(merged_coordinates), T + 1))

    # Iterate over scales
    for scale_idx in range(num_scales):

        # Iterate over orientations
        for rotation_idx in range(num_rotations):

            # Convolve image with both phase shifts at once,
            # as the real and the imaginary part of a complex kernel
            kernel = kernels[0, scale_idx, rotation_idx] + 1j * kernels[1, scale_idx, rotation_idx]
            R_complex = scipy.fft.ifft2(x_spectrum * scipy.fft.fft2(kernel, fft_shape))
            R_complex = R_complex[7:height, 7:width]  # valid part

            for phase_idx, R in enumerate([R_complex.real, R_complex.imag]):

                # Quantization
                R = np.abs(tools.matlab.round(R / quantization_steps[scale_idx]))

                # Truncation
                R = np.minimum(R, T).astype(int)

                # Histogram of each phase in the 8x8 grid, of shape [64, T + 1]
                counts = np.bincount((phases * (T + 1) + R).flatten(), minlength=64 * (T + 1))
                counts = counts.reshape(64, T + 1)

                # Feature extraction and merging, similar to DCTR features
                # merged_coordinates contains a list of lists, where the inner list contains between 0 and 4 coordinate tuples
                for hist_idx, coordinates in enumerate(merged_coordinates):
                    sub_features = sum(counts[row_shift * 8 + col_shift] for row_shift, col_shift in coordinates)
                    intermediate_features[phase_idx, scale_idx, rotation_idx, hist_idx, :] = sub_features / np.sum(sub_features)

    # Prepare for merging histograms with identical scale and symmetrical direction
    # Examples: theta = pi / 8 and 7 pi / 8, as well as theta = 2 pi / 8 and 6 pi / 8
    # The justification is that images have similar statistical characteristics in symmetrical orientations.
    # To my understanding, the assumption is that images are invariant to flipping.

    # At this point, we have 2 phase shifts * 32 rotations * 4 scales * 25 histograms * (T=4 + 1) values.
    # Now we merge S rotations into (S/2 + 1) values.

    # Dimensions of merged features
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
    merged_features = np.zeros(final_shape, dtype=float)

    # Copy features for orientation 0
    merged_features[:, :, 0, :, :] = intermediate_features[:, :, 0, :, :]

    # Copy features for orientation pi / 2
    merged_features[:, :, -1, :, :] = intermediate_features[:, :, num_rotations // 2, :, :]

    # Merge orientations i / pi and (pi - i) / pi
    rotation_indices = np.arange(1, num_rotations // 2)
    merged_features[:, :, rotation_indices, :, :] = (intermediate_features[:, :, rotation_indices, :, :] + intermediate_features[:, :, -rotation_indices, :, :]) / 2

    # One histogram per key, in the order of the dimensions
    features = OrderedDict()
    for idx in np.ndindex(final_shape[:-1]):
        features['_'.join(map(str, idx))] = merged_features[idx]
    return features


def extract(
    x: np.ndarray = None,
    *,
    num_rotations: int = 32,
    quantization_steps: Sequence[float] = (2, 4, 6, 8),
    T: int = 4,
    implementation: Implementation = Implementation.GFR_ORIGINAL,
    img: np.ndarray = None,
) -> OrderedDict:
    """
    Extract the Gabor filter residual features from a given image.

    :param x: decompressed grayscale image of shape [height, width], in [0, 255]
    :type x: np.ndarray
    :param num_rotations: number of rotations for Gabor kernel
    :type num_rotations: int
    :param quantization_steps: quantization step for each of the four scales.
        The default is for JPEG quality factor 75, use [0.1, 1, 1.5, 2] for quality factor 95.
    :type quantization_steps: list of float
    :param T: the highest histogram bin value after quantization. The histogram contains T + 1 bins corresponding to the values [0, ..., T]. Quantized values exceeding T will be clamped to T.
    :type T: int
    :param implementation:
    :type implementation: `Implementation`
    :param img: deprecated, use x
    :type img: np.ndarray
    :return: extracted Gabor features, one histogram of T + 1 values per key "{phase}_{scale}_{rotation}_{hist}",
        indexing the phase shifts, scales, merged rotations (num_rotations / 2 + 1) and merged DCTR phases (25).
        Use sw.tools.flatten() to obtain a 1D feature descriptor.
    :rtype: collections.OrderedDict

    :Examples:

    By default, this function uses the Rust implementation. To use the Python implementation, type

    >>> with sw.BACKEND_PYTHON:
    >>>     features = sw.gfr.extract(x)
    """
    if img is not None:
        warnings.warn('parameter img is deprecated, use x', DeprecationWarning, stacklevel=2)
        if x is not None:
            raise TypeError('both x and img given')
        x = img
    if x is None:
        raise TypeError('x must be given')
    if len(quantization_steps) != 4:
        raise ValueError('Expected four quantization steps, one for each scale')
    # Set up number of orientations from [0, 180) degree.
    # The number of rotations is denoted as S in the paper.
    if num_rotations % 2 != 0:
        raise ValueError('Expected number of rotations to be an even number')

    # choose implementation, Python if Rust is not available or does not support the input
    backend = tools.get_backend()
    rs = tools.rust_module('gfr')
    rust_supported = min(np.shape(x)) >= 8
    if backend == tools.BACKEND_RUST and rs is not None and rust_supported:
        return OrderedDict(rs.extract(
            x=np.ascontiguousarray(x, dtype=np.float64),
            kernels=compute_gabor_kernels(num_rotations, implementation=implementation),
            quantization_steps=np.array(quantization_steps, dtype=np.float64),
            T=T,
        ))
    elif backend in {tools.BACKEND_RUST, tools.BACKEND_PYTHON}:
        return _extract(
            x,
            num_rotations=num_rotations,
            quantization_steps=quantization_steps,
            T=T,
            implementation=implementation,
        )
    else:
        raise NotImplementedError(f'unknown backend {backend}')


def extract_from_file(
    path: Union[Path, str],
    num_rotations: int = 32,
    qf: int = None,
    quantization_steps: Sequence[float] = None,
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
    :param implementation:
    :return: extracted Gabor features, see extract()
    :rtype: collections.OrderedDict
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
        x=img,
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
