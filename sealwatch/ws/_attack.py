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
    x1: np.ndarray,
    pixel_predictor: typing.Tuple[str, typing.Callable] = 'KB',
    correct_bias: bool = False,
    weighted: bool = True
) -> float:
    """
    Runs weighted stego-image (WS) steganalysis on a given image.

    The goal of WS steganalysis is to estimate the embedding rate of uniform LSB replacement embedding.

    :param x1:
    :type x1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
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
    if len(x1.shape) == 2:
        x1 = x1[..., None]

    # cover with LSB flipped (anti-stego)
    x1_bar = x1 ^ 1

    # convert spatial to float
    x1 = x1.astype('float32')

    # estimate pixel value from its neighbors
    if not callable(pixel_predictor):
        pixel_predictor = get_pixel_predictor(pixel_predictor)
    x0_hat_ = pixel_predictor(x1)

    # compute weights
    if weighted:
        # estimate local variance
        avg = PIXEL_PREDICTORS['AVG9']
        mu = scipy.signal.convolve(x1[..., :1], avg[..., ::-1], mode='valid')
        mu2 = scipy.signal.convolve(x1[..., :1]**2, avg[..., ::-1], mode='valid')
        var = mu2 - mu**2

        # weight flat areas more
        weights = 1 / (5 + var)
        weights = weights / np.sum(weights)

    # unweighted - all areas equal
    else:
        weights = np.ones_like(x0_hat_) / x0_hat_.size

    # crop to match convolutions with valid padding
    x1_ = x1[1:-1, 1:-1, :1]
    x1_bar_ = x1_bar[1:-1, 1:-1, :1]

    # estimate payload
    try:
        beta_hat = np.sum(
            weights * (x1_ - x1_bar_) * (x1_ - x0_hat_),
        )
        beta_hat = np.clip(beta_hat, 0, None)
        # print(f'beta: {beta_hat} [{alpha/2 if not np.isnan(alpha) else 0}]')
    except ValueError:
        raise
        beta_hat = None

    # compute bias
    if correct_bias:
        spatial1_bias = pixel_predictor(x1_bar - x1)
        beta_hat -= beta_hat * np.sum(
            weights * (x1_ - x1_bar_) * spatial1_bias
        )
    return beta_hat
