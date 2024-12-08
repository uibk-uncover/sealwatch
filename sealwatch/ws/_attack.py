"""Implementation of WS steganalysis.

Based on SPIE, 2004 paper:
Jessica Fridrich, Miroslav Goljan,
On estimation of secret message length in LSB steganography in spatial domain

Inspired by implementation by Rémi Cogranne.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import scipy.signal
import typing


PIXEL_PREDICTORS = {
    'KB': np.array([[
        [-1, +2, -1],
        [+2,  0, +2],
        [-1, +2, -1],
    ]], dtype='float32').T / 4.,
    'AVG': np.array([[
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]], dtype='float32').T / 8.,
    'AVG9': np.array([[
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]], dtype='float32').T / 9.,
}


def get_pixel_predictor(predictor: str) -> np.ndarray:
    kernel = PIXEL_PREDICTORS[predictor]

    def predict(x):
        return scipy.signal.convolve(
            x.astype('float32'), kernel[..., ::-1],
            mode='valid',
        )[..., :1]

    return predict


def attack(
    spatial: np.ndarray,
    pixel_predictor: typing.Tuple[str, typing.Callable] = 'KB',
    correct_bias: bool = False,
    weighted: bool = True
) -> float:
    """
    Runs weighted stego-image (WS) steganalysis on a given image.

    The goal of WS steganalysis is to estimate the embedding rate of uniform LSB replacement embedding.

    :param spatial:
    :type spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param pixel_predictor:
    :type pixel_predictor:
    :param correct_bias:
    :type correct_bias: bool
    :param weighted:
    :type weighted: bool
    :return: change rate estimate
    :rtype: float
    """
    # add channel axis
    if len(spatial.shape) == 2:
        spatial = spatial[..., None]

    # cover with LSB flipped (anti-stego)
    spatial_bar = spatial ^ 1

    # convert spatial to float
    spatial = spatial.astype('float32')

    # estimate pixel value from its neighbors
    if not callable(pixel_predictor):
        pixel_predictor = get_pixel_predictor(pixel_predictor)
    spatial1_hat = pixel_predictor(spatial)

    # compute weights
    if weighted:
        # estimate local variance
        avg = PIXEL_PREDICTORS['AVG9']
        mu = scipy.signal.convolve(spatial[..., :1], avg[..., ::-1], mode='valid')
        mu2 = scipy.signal.convolve(spatial[..., :1]**2, avg[..., ::-1], mode='valid')
        var = mu2 - mu**2

        # weight flat areas more
        weights = 1 / (5 + var)
        weights = weights / np.sum(weights)

    # unweighted - all areas equal
    else:
        weights = np.ones_like(spatial1_hat) / spatial1_hat.size

    # crop to match convolutions with valid padding
    spatial1 = spatial[1:-1, 1:-1, :1]
    spatial1_bar = spatial_bar[1:-1, 1:-1, :1]

    # estimate payload
    try:
        beta_hat = np.sum(
            weights * (spatial1 - spatial1_bar) * (spatial1 - spatial1_hat),
        )
        beta_hat = np.clip(beta_hat, 0, None)
        # print(f'beta: {beta_hat} [{alpha/2 if not np.isnan(alpha) else 0}]')
    except ValueError:
        beta_hat = None

    # compute bias
    if correct_bias:
        spatial1_bias = pixel_predictor(spatial_bar - spatial)
        beta_hat -= beta_hat * np.sum(
            weights * (spatial1 - spatial1_bar) * spatial1_bias
        )
    return beta_hat

    # # 3x3 box filter
    # kernel_avg = np.ones((3, 3)) / 3**2

    # # estimate per channel
    # # alpha_hat = 0.
    # alphas_hat = []
    # for ch in range(img.shape[2]):

    #     # Select channel
    #     x = img[:, :, ch]

    #     # Cast to float before convolution
    #     x = x.astype(float)

    #     # Estimate cover pixel value from its neighbors
    #     x_hat = scipy.signal.convolve2d(x, kernel_estimator, 'valid')

    #     # Compute variance
    #     mu = scipy.signal.convolve2d(x, kernel_avg, 'valid')
    #     mu2 = scipy.signal.convolve2d(x**2, kernel_avg, 'valid')
    #     var = mu2 - mu**2
    #     weights = 1 / (4 + var)
    #     weights = weights / np.sum(weights)

    #     # crop to match convolutions with valid padding
    #     x = x[1:-1, 1:-1]
    #     x_bar = x_bar[1:-1, 1:-1]

    #     # estimate payload
    #     alpha_hat = np.sum(2 * weights * (x - x_bar) * (x - x_hat))
    #     alphas_hat.append(alpha_hat)

    # # aggregate alphas by adding the channel estimates :)
    # # return np.sum(alphas_hat)
    # return alphas_hat

