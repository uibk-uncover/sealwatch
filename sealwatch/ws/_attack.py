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

    Introduced in
    J. Fridrich, M. Goljan.
    On estimation of secret message length in LSB steganography in spatial domain.
    Security, Steganography, and Watermarking of Multimedia Contents VI, Proc. SPIE, vol. 5306, 2004.

    Revisited in
    A. D. Ker, R. Böhme.
    Revisiting weighted stego-image steganalysis.
    Security, Forensics, Steganography, and Watermarking of Multimedia Contents X, Proc. SPIE, vol. 6819, 2008.

    :param x1: grayscale image of shape [height, width], or color image of shape [height, width, channels],
        integers in [0, 255]
    :type x1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param pixel_predictor: predictor of each pixel from its neighbors,
        'KB' for the predictor of Ker and Böhme, 'AVG' for the average of the 8 neighbors,
        or a callable, taking the image of shape [height, width, channels]
        and returning the prediction of shape [height - 2, width - 2, 1],
        e.g., the U-Net of unet_estimator()
    :type pixel_predictor: str or callable
    :param correct_bias: whether to correct the bias of the estimate, as of Ker and Böhme
    :type correct_bias: bool
    :param weighted: whether to weight the pixels by their local variance, flat areas more
    :type weighted: bool
    :return: estimated change rate, i.e., half of the embedding rate of LSB replacement
    :rtype: float

    :Example:

    >>> x1 = np.array(Image.open("suspicious.png").convert("L"))
    >>> beta_hat = sw.ws.attack(x1)  # change rate, about 0 for cover

    With the U-Net pixel predictor, for grayscale images of 512x512 pixels,

    >>> predictor = sw.ws.unet_estimator()  # downloads the weights once
    >>> beta_hat = sw.ws.attack(x1, pixel_predictor=predictor)
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
