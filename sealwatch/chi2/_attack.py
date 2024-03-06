"""Implementes LSB histogram attack with chi-square test.

From Westfeld, Pfitzmann: Attacks on Steganographic Systems.
https://users.ece.cmu.edu/~adrian/487-s06/westfeld-pfitzmann-ihw99.pdf

Inspired by Remi Cogranne's implementation.

The attack can be used against nearly-fully embedded image with LSB replacement.

Authors: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import logging
import numpy as np
import scipy.stats
import typing

from ..tools import EPS


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


# def get_path(
#     img: np.ndarray,
#     seed: int = None,
#     generator: str = None,
# ) -> typing.Tuple[np.ndarray]:
#     """Generates path through image from seed.

#     Args:
#         img (np.ndarray): Pixel tensor.
#         seed (int): Random number generator seed for reproducibility.
#         generator (str): Random number generator. One of None (numpy default), or MT19937, used by Matlab.
#     Returns:
#         (tuple): Indices through array.

#     Example:
#         Sequential pass, channel-column-row.
#         >>> idx = revelio.chi2.get_path(img)
#         >>> img[idx].reshape(img)

#         Sequential pass, channel-row-column.
#         >>> idx = revelio.chi2.get_path(img.transpose(1, 0, 2))
#         >>> idx = tuple([idx[i] for i in [1, 0, 2]])

#         Permuted pass.
#         >>> idx = revelio.chi2.get_path(img, seed=12345)
#     """
#     # sequential path
#     perm = np.arange(img.size)
#     # permutative straddling
#     if seed is not None:
#         # set generator
#         if generator is None:  # default
#             rng = np.random.default_rng(seed)
#         elif generator == 'MT19937':  # Matlab
#             rng = np.random.RandomState(seed)
#         else:
#             raise NotImplementedError(f'unsupported generator {generator}')
#         # permute
#         perm = rng.permutation(perm)
#     # permutation to indices
#     idx = []
#     for dim in reversed(img.shape):
#         idx.append(perm % dim)
#         perm //= dim
#     return tuple(reversed(idx))


# def attack_along_path(
#     img: np.ndarray,
#     message_lengths: np.ndarray,
#     generator: str = None,
#     seed: int = None,
# ) -> typing.Tuple[np.ndarray]:
#     """Chi2 test for testing a specific path.

#     Calls `attack()` with subsets of image along certain path.
#     Returns scores in a compact way.
#     This way, one can test

#     Coded based on discussion with Jan Butora,
#     inspired by Remi Cogranne's implementation.

#     Args:
#         x (np.ndarray): Spatial domain, cover or stego.
#         message_lengths (np.ndarray): Grid of message lengths to test.
#         seed (int): Random path. If None, sequential pass is used.
#         generator (str): Random number generator. One of None (numpy default), or MT19937, used by Matlab.
#     Returns:
#         (np.ndarray): Chi2 scores corresponding to M-grid.
#     """
#     # select path
#     path_permutation = get_path(img=img, seed=seed, generator=generator)

#     # scores per message length
#     scores, pvalues = [], []
#     for message_length in message_lengths:
#         logging.info(f'running chi2 along {seed} for {message_length}')

#         # attack first m_i image pixels
#         path_permutation_i = tuple([p[:message_length] for p in path_permutation])
#         score_i, pvalue_i = attack(img[path_permutation_i])
#         scores.append(score_i)
#         pvalues.append(pvalue_i)

#     return np.array(scores), np.array(pvalues)
