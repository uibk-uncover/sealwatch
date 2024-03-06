"""Implementes LSB histogram attack with chi-square test.

From Westfeld, Pfitzmann: Attacks on Steganographic Systems.
https://users.ece.cmu.edu/~adrian/487-s06/westfeld-pfitzmann-ihw99.pdf

Inspired by Remi Cogranne's implementation.

The attack can be used against nearly-fully embedded image with LSB replacement.

Authors: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
import scipy.stats
import typing


def attack(
    spatial: np.ndarray,
) -> typing.Tuple[float]:
    """Measures the "distance" between the observed histogram and a typical histogram after LSB replacement.

    LSB replacement (embedding rate = 1) averages the neighboring histogram bins.

    :param spatial: image pixels, of arbitrary shape
    :type spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :returns: distance and p-value,
        distance is the chi2 test statistic between the observed histogram and the stego model.
        A small distance means that the image matches the model (e.g., because it was embedded with LSB replacement).
        The p-value turns the score into a probability.
        A p-value of 0 means that the image contains no steganography.
        p-value of 1 indicates that the image contains LSBR steganography.
    """
    # Calculate histogram
    h, bin_edges = np.histogram(spatial.flatten(), bins=np.arange(0, 2**8+1))

    # Average neighboring pairs, and repeat the result.
    # This is the distribution that LSB replacement would produce.
    h_pair_avg = np.repeat((h[:-1:2] + h[1::2]) / 2, 2)

    # Keep only values where h[bin_idx] > 4 (recommended in Jessica Fridrich's book, appendix D).
    # This also prevents division by zero.
    mask = h_pair_avg > 4
    h = h[mask]
    h_pair_avg = h_pair_avg[mask]

    # Measure the distance between the observed histogram and our model of stego images.
    # This is our test statistic for the chi-square test.
    distance = np.sum((h - h_pair_avg)**2 / h_pair_avg)

    # If the image matches the model, the distance should follow the chi-squared distribution.
    degrees_of_freedom = 2 ** 8 - 1
    p_value = scipy.stats.chi2.sf(distance, degrees_of_freedom)

    return distance, p_value
