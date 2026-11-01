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

    Introduced in
    A. Westfeld, A. Pfitzmann.
    Attacks on steganographic systems: Breaking the steganographic utilities EzStego, Jsteg, Steganos,
    and S-Tools - and some lessons learned.
    Information Hiding, Lecture Notes in Computer Science, vol. 1768, 2000.

    :param spatial: image pixels of arbitrary shape, integers in [0, 255]
    :type spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: chi2 distance to the histogram after LSB replacement, small for stego,
        and its p-value, close to 1 after LSB replacement of all pixels
    :rtype: tuple of float
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
