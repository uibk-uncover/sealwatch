"""
Implementation of the PHARM features as described in

V. Holub and J. Fridrich, Phase-Aware Projection Model for Steganalysis of JPEG Images,
Proc. SPIE, Electronic Imaging, Media Watermarking, Security, and Forensics XVII, vol. 9409, 2015.
http://dde.binghamton.edu/vholub/pdf/SPIE15_Phase-Aware_Projection_Model_for_Steganalysis_of_JPEG_Images.pdf

Authors: Benedikt Lorch, Martin Benes
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
from typing import Iterator, List, Tuple, Union
import warnings

from .. import tools


class Implementation(enum.Enum):
    """PHARM implementation to choose from."""

    PHARM_ORIGINAL = 1
    """Original PHARM implementation by DDE."""
    PHARM_FIX = 2
    """PHARM implementation with fixes.

    It deviates from the original Matlab implementation in two aspects:
        - It fixes a bug in the symmetrization.
        - It does not crop the image borders in order to simplify the indices.
          This is why it produces different results even when the symmetrization is disabled.
    """
    PHARM_REVISITED = 2
    """Legacy name of PHARM_FIX."""


def qf_to_quantization_step(
    qf: int,
) -> float:
    """Quantization step for a JPEG quality factor."""
    return (65 / 4) - (3 / 20) * qf


def select_quantization_step(
    path: Union[str, Path],
) -> float:
    """Quantization step for the quality factor of a JPEG file."""
    qf = tools.jpeg.identify_qf(path)
    return qf_to_quantization_step(qf)


def _residuals(
    x: np.ndarray,
    implementation: Implementation,
) -> List[Tuple[str, np.ndarray, Tuple[int, int]]]:
    """Noise residuals of the image, in the order of the submodels.

    :return: list of (name, residual, kernel shape).
        For PHARM_ORIGINAL, the kernel shape is the size of the border cropped by the Matlab implementation.
    """
    if implementation is Implementation.PHARM_ORIGINAL:
        height, width = x.shape
        # Cut off one (or two) pixels at all sides in order to avoid out-of-bounds accesses
        I, J = slice(1, height - 1), slice(1, width - 1)
        I_, J_ = slice(I.start - 1, I.stop - 1), slice(J.start - 1, J.stop - 1)  # up, left
        I2, J2 = slice(2, height - 2), slice(2, width - 2)
        return [
            # First order kernels: [-1, 1] in horizontal and vertical directions; the implementation swaps the sign of U
            ('s1_spam14_R', -x[I, J] + x[I, J.start + 1:J.stop + 1], (3, 3)),
            ('s1_spam14_U', x[I_, J] - x[I, J], (3, 3)),
            # Third order kernels: [1, -3, 3, -1] in horizontal and vertical directions; the implementation swaps the sign of U
            ('s3_spam14_R', (
                + x[I2, J2.start - 1:J2.stop - 1] - 3 * x[I2, J2]
                + 3 * x[I2, J2.start + 1:J2.stop + 1] - x[I2, J2.start + 2:J2.stop + 2]
            ), (5, 5)),
            ('s3_spam14_U', (
                - x[I2.start - 2:I2.stop - 2, J2] + 3 * x[I2.start - 1:I2.stop - 1, J2]
                - 3 * x[I2, J2] + x[I2.start + 1:I2.stop + 1, J2]
            ), (5, 5)),
            # Second order 2x2 kernels; the implementation swaps the signs of V and DMaj
            # [[ 1,  1],
            #  [-1, -1]]
            ('s2x2_spam14_H', + x[I_, J_] + x[I_, J] - x[I, J_] - x[I, J], (3, 3)),
            # [[-1,  1],
            #  [-1,  1]]
            ('s2x2_spam14_V', + x[I_, J_] - x[I_, J] + x[I, J_] - x[I, J], (3, 3)),
            # [[ 1, -1],
            #  [-1,  1]]
            ('s2x2_spam14_DMaj', - x[I_, J_] + x[I_, J] + x[I, J_] - x[I, J], (3, 3)),
        ]
    elif implementation is Implementation.PHARM_FIX:
        # residuals aligned with the top-left corner of the image
        return [
            # First order kernels: [-1, 1] in horizontal and vertical directions
            ('s1_spam14_R', -x[:, :-1] + x[:, 1:], (1, 2)),
            ('s1_spam14_D', -x[:-1, :] + x[1:, :], (2, 1)),
            # Third order kernels: [1, -3, 3, -1] in horizontal and vertical directions
            ('s3_spam14_R', + x[:, :-3] - 3 * x[:, 1:-2] + 3 * x[:, 2:-1] - x[:, 3:], (1, 4)),
            ('s3_spam14_D', + x[:-3, :] - 3 * x[1:-2] + 3 * x[2:-1, :] - x[3:, :], (4, 1)),
            # Second order 2x2 kernels
            # [[ 1,  1],
            #  [-1, -1]]
            ('s2x2_spam14_H', + x[:-1, :-1] + x[:-1, 1:] - x[1:, :-1] - x[1:, 1:], (2, 2)),
            # [[-1,  1],
            #  [-1,  1]]
            ('s2x2_spam14_V', - x[:-1, :-1] + x[:-1, 1:] - x[1:, :-1] + x[1:, 1:], (2, 2)),
            # [[ 1, -1],
            #  [-1,  1]]
            ('s2x2_spam14_DMaj', + x[:-1, :-1] - x[:-1, 1:] - x[1:, :-1] + x[1:, 1:], (2, 2)),
        ]
    else:
        raise NotImplementedError(f'unknown implementation {implementation}')


def _flipped_offset(
    shift: int,
    projection_size: int,
    kernel_size: int,
    implementation: Implementation,
) -> int:
    """Offset of the flipped projection matrix, so that it covers the flipped DCT grid phases.

    :param shift: offset of the projection matrix
    :param projection_size: size of the projection matrix along the axis
    :param kernel_size: kernel size along the axis, see _residuals()
    """
    if implementation is Implementation.PHARM_ORIGINAL:
        # Shift is one lower than in Matlab
        return (kernel_size - 1 - shift - projection_size) % 8
    else:
        return (7 - shift - (projection_size - 1) - (kernel_size - 1)) % 8


def _histogram(
    projection: np.ndarray,
    q: float,
    T: int,
) -> np.ndarray:
    """Histogram of the absolute projections in bins [0, q), [q, 2q), ..., [(T-1)q, Tq],
    larger values are dropped. Equal to np.histogram(np.abs(projection), np.arange(T + 1) * q)[0]."""
    a = np.abs(projection)
    edges = np.arange(T + 1) * q
    # number of values below each edge, the last one inclusive
    below = [np.count_nonzero(a < e) for e in edges[1:-1]] + [np.count_nonzero(a <= edges[-1])]
    return np.diff(below, prepend=0)


def _strided_histograms(
    residual: np.ndarray,
    kernels: List[Tuple[np.ndarray, int, int]],
    q: float,
    T: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Histograms of the strided convolutions of the residual with each kernel, one at a time.

    :param kernels: list of (kernel, offset_y, offset_x)
    :return: histograms of shape [len(kernels), T], and the number of convolution outputs of each kernel
    """
    counts, sizes = [], []
    for kernel, offset_y, offset_x in kernels:
        proj = tools.signal.strided_convolution(
            img=residual,
            kernel=kernel,
            stride=8,
            offset_y=offset_y,
            offset_x=offset_x,
        )
        counts.append(_histogram(proj, q, T))
        sizes.append(proj.size)
    return np.array(counts), np.array(sizes)


def _strided_convolutions_batched(
    residual: np.ndarray,
    kernels: List[Tuple[np.ndarray, int, int]],
    batch_size: int = 256,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Strided convolutions of the residual with batches of kernels, as matrix products.

    Each output of a convolution with stride 8, with a kernel of at most 8x8 at an offset of at most 7,
    lies within the 15x15 patch at an 8x8 block. Embedding each kernel at its offset in a 15x15 kernel,
    the convolutions of all kernels are a product of the patches and the embedded kernels.

    The products sum in a different order than strided_convolution(),
    so that the outputs differ in the order of 1e-13.

    :param kernels: list of (kernel, offset_y, offset_x)
    :return: per batch, the outputs of shape [blocks_y, blocks_x, batch],
        and the mask of the outputs within the residual, as in strided_convolution()
    """
    H, W = residual.shape
    blocks_y, blocks_x = (H + 7) // 8, (W + 7) // 8
    # 15x15 patches at each 8x8 block, of shape [blocks, 225]
    padded = np.zeros((blocks_y * 8 + 7, blocks_x * 8 + 7))
    padded[:H, :W] = residual
    patches = np.lib.stride_tricks.sliding_window_view(padded, (15, 15))[::8, ::8].reshape(blocks_y * blocks_x, 225)

    for batch in range(0, len(kernels), batch_size):
        embedded = np.zeros((15, 15, len(kernels[batch:batch + batch_size])))
        valid = np.zeros((blocks_y, blocks_x, embedded.shape[-1]), dtype=bool)
        for i, (kernel, offset_y, offset_x) in enumerate(kernels[batch:batch + batch_size]):
            kernel_height, kernel_width = kernel.shape
            # convolution, as correlation with the flipped kernel
            embedded[offset_y:offset_y + kernel_height, offset_x:offset_x + kernel_width, i] = kernel[::-1, ::-1]
            # outputs within the residual
            valid[:1 + (H - offset_y - kernel_height) // 8, :1 + (W - offset_x - kernel_width) // 8, i] = True
        yield (patches @ embedded.reshape(225, -1)).reshape(valid.shape), valid


def _strided_histograms_batched(
    residual: np.ndarray,
    kernels: List[Tuple[np.ndarray, int, int]],
    q: float,
    T: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as _strided_histograms(), with the convolutions as matrix products, several times faster.

    As the convolution outputs differ in the order of 1e-13, see _strided_convolutions_batched(),
    a histogram may differ, if an output lies that close to a bin edge.
    """
    counts, sizes = [], []
    for proj, valid in _strided_convolutions_batched(residual, kernels):
        # histograms as in _histogram(), outputs outside the residual fall into no bin
        a = np.where(valid, np.abs(proj), np.inf)
        edges = np.arange(T + 1) * q
        below = [np.count_nonzero(a < e, axis=(0, 1)) for e in edges[1:-1]] + [np.count_nonzero(a <= edges[-1], axis=(0, 1))]
        counts.append(np.diff(below, axis=0, prepend=0).T)
        sizes.append(np.count_nonzero(valid, axis=(0, 1)))
    return np.concatenate(counts), np.concatenate(sizes)


def _projection_histograms(
    residual: np.ndarray,
    kernel_shape: Tuple[int, int],
    rng: np.random.RandomState,
    *,
    implementation: Implementation,
    q: float,
    T: int,
    num_projections: int,
    maximum_projection_size: int,
    symmetrize: bool,
    normalize: bool,
    batched: bool,
) -> np.ndarray:
    """Histograms of random projections of a residual, of shape [num_projections, T]."""
    # projection matrices (kernels) at their offsets, the flipped ones merged into the same histogram
    kernels, kernel_projections = [], []
    for proj_idx in range(num_projections):
        # Randomly select height and width of projection matrix, each between [1, s]
        proj_mat_shape = tools.matlab.randi(low=1, high=maximum_projection_size + 1, size=2, rng=rng)

        # Randomly select grid offset
        shift_y, shift_x = tools.matlab.randi(low=0, high=8, size=2, rng=rng)

        # Generate random projection matrix. Reshape to match Matlab order
        proj_mat = tools.matlab.randn(proj_mat_shape, rng=rng).flatten(order="C").reshape(proj_mat_shape, order="F")

        # Normalize so that the Frobenius norm of the projection matrix is 1
        proj_mat = proj_mat / np.sqrt(np.sum(proj_mat ** 2))

        # (1) Original orientation
        projections = [(proj_mat, shift_y, shift_x)]
        if symmetrize:
            # Merge with the histograms of the flipped projection matrices at the flipped grid phases
            flipped_y = _flipped_offset(shift_y, proj_mat_shape[0], kernel_shape[0], implementation)
            flipped_x = _flipped_offset(shift_x, proj_mat_shape[1], kernel_shape[1], implementation)
            projections += [
                (np.flipud(proj_mat), flipped_y, shift_x),  # (2) Vertical kernel flip
                (np.fliplr(proj_mat), shift_y, flipped_x),  # (3) Horizontal kernel flip
                (np.flipud(np.fliplr(proj_mat)), flipped_y, flipped_x),  # (4) Both horizontal and vertical kernel flip
            ]
        kernels += projections
        kernel_projections += [proj_idx] * len(projections)

    if batched:
        counts, sizes = _strided_histograms_batched(residual, kernels, q, T)
    else:
        counts, sizes = _strided_histograms(residual, kernels, q, T)

    h = np.zeros((num_projections, T), dtype=float if normalize else int)
    for proj_idx, hist, size in zip(kernel_projections, counts, sizes):
        h[proj_idx, :] += hist / size if normalize else hist
    return h


def extract(
    x: np.ndarray = None,
    *,
    implementation: Implementation = Implementation.PHARM_FIX,
    q: float = 5,
    T: int = 2,
    num_projections: int = 900,
    maximum_projection_size: int = 8,
    first_order_residuals: bool = True,
    second_order_residuals: bool = True,
    third_order_residuals: bool = True,
    symmetrize: bool = True,
    normalize: bool = False,
    seed: int = 1,
    batched: bool = True,
    x1: np.ndarray = None,
) -> OrderedDict:
    """Extracts the PHARM features from a given decompressed image.

    The PHARM features were introduced in
    V. Holub and J. Fridrich, Phase-Aware Projection Model for Steganalysis of JPEG Images.
    SPIE Electronic Imaging, Media Watermarking, Security, and Forensics XVII, vol. 9409, 2015.
    http://dde.binghamton.edu/vholub/pdf/SPIE15_Phase-Aware_Projection_Model_for_Steganalysis_of_JPEG_Images.pdf

    :param x: decompressed JPEG image
        of shape [height, width]
    :type x: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param implementation: implementation of PHARM to use
    :type implementation: `Implementation`
    :param q: quantization step, see qf_to_quantization_step()
    :type q: float
    :param T: truncation threshold
    :type T: int
    :param num_projections: number of random projection matrices, 900 as in the original implementation.
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
    :param seed: seed for random number generator for the projection matrices.
        PHARM_ORIGINAL seeds each submodel by seed + its index, PHARM_FIX draws the submodels in sequence,
        so its projection matrices depend on which residuals are included.
    :type seed: int
    :param batched: If True, compute the projections of each residual as matrix products, several times faster.
        The projections differ in the order of 1e-13, so that rarely, a histogram count may differ.
        If False, compute one projection at a time, bit-exact to the former implementation.
    :type batched: bool
    :param x1: deprecated, use x
    :type x1: np.ndarray
    :return: features as ordered dictionary, where the keys are the submodel names and the values are the features of shape [num_projections, T].
        Unless normalize is True, the features are not normalized.
    :rtype: `OrderedDict`

    :Example:

    >>> features = sw.pharm.extract(x, implementation=sw.PHARM_ORIGINAL, num_projections=900)
    """
    if x1 is not None:
        warnings.warn('parameter x1 is deprecated, use x', DeprecationWarning, stacklevel=2)
        if x is not None:
            raise TypeError('both x and x1 given')
        x = x1
    if x is None:
        raise TypeError('x must be given')

    # submodels: s1 (2 residuals), s3 (2 residuals), s2x2 (3 residuals)
    included = [first_order_residuals] * 2 + [third_order_residuals] * 2 + [second_order_residuals] * 3

    # PHARM_FIX draws all submodels from a single generator
    rng = np.random.RandomState(seed)

    features = OrderedDict()
    for idx, (name, residual, kernel_shape) in enumerate(_residuals(x, implementation)):
        if not included[idx]:
            continue
        if implementation is Implementation.PHARM_ORIGINAL:
            # PHARM_ORIGINAL seeds each submodel separately
            rng = np.random.RandomState(seed + idx)
        features[name] = _projection_histograms(
            residual,
            kernel_shape,
            rng,
            implementation=implementation,
            q=q,
            T=T,
            num_projections=num_projections,
            maximum_projection_size=maximum_projection_size,
            symmetrize=symmetrize,
            normalize=normalize,
            batched=batched,
        )
    return features


def extract_from_file(
    path: Union[str, Path],
    *,
    qf: int = None,
    q: float = None,
    **kw,
) -> OrderedDict:
    """Extracts the PHARM features from a given JPEG image.

    :param path: path to JPEG image
    :type path: str or `Path`
    :param qf: JPEG quality factor, used to select the quantization step
    :type qf: int
    :param q: quantization step. If neither q nor qf is given, q is 5, the quantization step of quality factor 75.
    :type q: float
    :param kw: other parameters of extract()
    :return: features as ordered dictionary, see extract()
    :rtype: `OrderedDict`

    :Example:

    >>> features = sw.pharm.extract_from_file('seal1.jpg', qf=75)
    """
    if q is None:
        q = 5 if qf is None else qf_to_quantization_step(qf)
    elif qf is not None:
        raise TypeError('both q and qf given')

    # Decompress image without rounding
    x = tools.jpeg.decompress_from_file(path)
    return extract(x, q=q, **kw)
