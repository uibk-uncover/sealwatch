"""Variance estimator of the multivariate Gaussian (MVG) cover model.

Copied from MiPOD in conseal, to keep sealwatch independent of conseal.

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
import scipy.fft
import scipy.signal


def wiener2(
    x: np.ndarray,
    kernel_size: int = 2,
    *,
    boundary: str = 'symm',
) -> np.ndarray:
    """Adaptive Wiener filter, compatible with Matlab's wiener2.

    :param x: 2D image
    :type x: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param kernel_size: size of the local neighborhood
    :type kernel_size: int
    :param boundary: padding of the local statistics, see
        `scipy.signal.correlate2d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html>`__;
        symmetric ('symm') by default, zero padding ('fill') as in Matlab
    :type boundary: str
    :return: filtered image
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    """
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    local_mean = scipy.signal.correlate2d(x, kernel, mode='same', boundary=boundary)
    local_mean_sq = scipy.signal.correlate2d(x**2, kernel, mode='same', boundary=boundary)
    local_variance = np.clip(local_mean_sq - local_mean**2, 1e-8, None)
    noise = np.mean(local_variance)
    gain = np.maximum(0, local_variance - noise) / np.maximum(local_variance, noise)
    return local_mean + gain * (x - local_mean)


def estimate_variance(
    x0: np.ndarray,
    *,
    block_size: int = 9,
    degree: int = 9,
    floor: float = .01,
    boundary: str = 'symm',
) -> np.ndarray:
    """Estimates the cover noise variance, as in MiPOD.

    Fits a local 2D-DCT polynomial model to the Wiener-filter residual
    (Eq. 21-26) and clips the variance from below (Eq. 27) of

    Sedighi, Cogranne, Fridrich.
    Content-Adaptive Steganography by Minimizing Statistical Detectability.
    IEEE TIFS, 2016.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param block_size: size of the local neighborhood, odd
    :type block_size: int
    :param degree: degree of the polynomial
    :type degree: int
    :param floor: minimum variance
    :type floor: float
    :param boundary: padding of the Wiener filter, see :func:`wiener2`
    :type boundary: str
    :return: estimated variance per pixel
        of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> v = sw.mvg.estimate_variance(x0)
    """
    assert len(x0.shape) == 2, 'grayscale expected'
    assert block_size % 2 == 1, 'block size must be odd'
    assert degree <= block_size, 'degree exceeds block size'
    x0 = x0.astype('float64')
    residual = x0 - wiener2(x0, boundary=boundary)

    # parametric model G, Eq. 25-26
    q = degree * (degree + 1) // 2
    base = np.zeros((block_size, block_size))
    base[0, 0] = 1
    G = np.zeros((block_size**2, q))
    k = 0
    for xshift in range(degree):
        for yshift in range(degree - xshift):
            basis = np.roll(base, shift=(xshift, yshift), axis=(0, 1))
            basis = scipy.fft.idct(scipy.fft.idct(basis, axis=0, norm='ortho'), axis=1, norm='ortho')
            G[:, k] = basis.flatten(order='F')
            k += 1

    # projection onto the orthogonal complement of G
    P_G_orth = np.eye(block_size**2) - G @ np.linalg.solve(G.T @ G, G.T)

    # sliding windows over the padded residual
    pad = block_size // 2
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(residual, pad, 'symmetric'),
        (block_size, block_size),
    ).reshape(-1, block_size**2).T

    # Eq. 24
    variance = np.sum((P_G_orth @ windows)**2, axis=0) / (block_size**2 - q)
    variance = variance.reshape(x0.shape)
    return np.clip(variance, floor, None)
