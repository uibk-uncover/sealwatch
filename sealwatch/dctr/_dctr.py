"""
Implementation of the DCTR features as described in

V. Holub and J. Fridrich
"Low-Complexity Features for JPEG Steganalysis Using Undecimated DCT"
IEEE Transactions on Information Forensics and Security
Vol. 10, No. 2, pp. 219-228, Feb. 2015
https://doi.org/10.1109/TIFS.2014.2364918

Author: Benedikt Lorch
Affiliation: University of Innsbruck

This implementation builds on the original Matlab implementation provided by the paper authors. Please find the license of the original implementation below.
-------------------------------------------------------------------------
Copyright (c) 2014 DDE Lab, Binghamton University, NY. All Rights Reserved.
Permission to use, copy, modify, and distribute this software for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that this copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from DDE Lab. DDE Lab does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall Binghamton University or DDE Lab be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software. DDE Lab disclaims any warranties, and has no obligations to provide maintenance, support, updates, enhancements or modifications.
-------------------------------------------------------------------------
"""  # noqa: E501


from collections import OrderedDict
import numpy as np
from pathlib import Path
import scipy.signal
from typing import Union, List

from .. import tools


def get_symmetric_histogram_coordinates() -> List[List]:
    """
    Pre-computes the locations which histograms can be merged based on the following symmetry considerations:
    - Convolve the spatial image with each of the 64 DCT basis functions. The result is called undecimated DCT.
    - The distribution of each individual mode is approximately symmetrical and centered at 0. Under this assumption, we can take the absolute value.
    - Exploiting the absolute values and the symmetry of projection vectors, the histograms can be merged.
    :return: list of lists of coordinates which modes of each undecimated DCT plane can be merged. The entries are as follows:

        [a] = (0, 0)
        [b] = (0, 1), (0, 7)
        [c] = (0, 2), (0, 6)
        [d] = (0, 3), (0, 5)
        [e] = (0, 4)
        [f] = (1, 0), (7, 0)
        [g] = (1, 1), (1, 7), (7, 1), (7, 7)
        [h] = (1, 2), (1, 6), (7, 2), (7, 6)
        [i] = (1, 3), (1, 5), (7, 3), (7, 5)
        [j] = (1, 4), (7, 4)
        [k] = (2, 0), (6, 0)
        [l] = (2, 1), (2, 7), (6, 1), (6, 7)
        [m] = (2, 2), (2, 6), (6, 2), (6, 6)
        [n] = (2, 3), (2, 5), (6, 3), (6, 5)
        [o] = (2, 4), (2, 6)
        [p] = (3, 0), (5, 0)
        [q] = (3, 1), (3, 7), (5, 1), (5, 7)
        [r] = (3, 2), (3, 6), (5, 2), (5, 6)
        [s] = (3, 3), (3, 5), (5, 3), (5, 5)
        [t] = (3, 4), (5, 4)
        [u] = (4, 0)
        [v] = (4, 1), (4, 7)
        [w] = (4, 2), (4, 6)
        [x] = (4, 3), (4, 5)
        [y] = (4, 4)

          | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7
        ---------------------------------
        0 | a   b   c   d   e   d   c   b
        1 | f   g   h   i   j   i   h   g
        2 | k   l   m   n   o   n   m   l
        3 | p   q   r   s   t   s   r   q
        4 | u   v   w   x   y   x   w   v
        5 | p   q   r   s   t   s   r   q
        6 | k   l   m   n   o   n   m   l
        7 | f   g   h   i   j   i   h   g
    """

    # Compute DCTR locations to be merged, see Table 1.
    merged_coordinates = []
    for i in range(5):
        for j in range(5):
            # Up to four histograms are merged for positions (a, b) in {1, 2, 3, 5, 6, 7}.
            # When exactly one element of (i, j) is in {0, 4}, only two histograms are merged.
            # When both elements (i, j) are in {0, 4}, there is only a single histogram.

            # The four candidate positions are (i, j), (i, 8-j), (8-i, j), (8-i, 8-j)
            coordinates = [
                (i, j),
                (i, 8 - j),
                (8 - i, j),
                (8 - i, 8 - j)
            ]

            # Drop coordinates that exceed the 8x8 grid
            coordinates = filter(lambda pair: pair[0] < 8 and pair[1] < 8, coordinates)

            # Drop duplicate coordinates
            coordinates = list(OrderedDict.fromkeys(coordinates).keys())

            # Save list
            merged_coordinates.append(coordinates)

    return merged_coordinates


def extract_from_file(
    path: Union[str, Path],
    qf: int,
) -> OrderedDict:
    """
    Extract DCTR features from the luminance channel of JPEG image given by its filepath

    :param path: path to JPEG image
    :type path: str
    :param qf: JPEG quality factor used to determine the quantization step
    :type qf:
    :return: DCTR features
        of shape [64x25, 5]
    :rtype:
    """

    # Decompress image to spatial domain
    x1 = tools.jpeg.decompress_from_file(path)

    # Undo level shift, in line with the Matlab implementation
    x1 -= 128

    # Compute quantization step based on quality factor
    if qf < 50:
        q = min(8 * (50 / qf), 100)
    else:
        # See Eq. 10
        q = max(8 * (2 - (qf / 50)), 0.2)

    return extract(x1=x1, q=q)


def extract(
    x1: np.ndarray,
    q: float,
    *,
    T: np.ndarray = 4,
) -> OrderedDict:
    """Extracts DCTR features from the provided image.

    Note that there can be minor differences during quantization, which is why the Matlab and Python results do not match perfectly.

    :param x1: grayscale image with intensities in range [-128, 127]
    :type x1:
    :param q: quantization step
    :type q: float
    :param T: truncation threshold. The number of histogram bins is T + 1.
    :type T:
    :return: DCTR features of shape [64x25, 5]
    :rtype:

    :Example:

    >>> # TODO
    """

    # Compute 2D DCT basis patterns as in Eq. 2
    dct_mat = tools.dct.compute_dct_mat()

    # Compute DCTR locations to be merged, see Table 1.
    merged_coordinates = get_symmetric_histogram_coordinates()

    # Allocate space for features
    features = np.zeros((64, len(merged_coordinates), T + 1,))
    features = OrderedDict()

    # The bins correspond to values [0, ..., T]
    # np.histogram requires bin edges
    # Bin 0: [-0.5, 0.5]
    # Bin 1: [ 0.5, 1.5]
    # ...
    # Bin T: [T-0.5, T+0.5]
    bin_edges = np.arange(T + 2) - 0.5

    for mode_row in np.arange(8):
        for mode_col in np.arange(8):
            # Linear index
            mode_idx = mode_row * 8 + mode_col

            # Get DCT base for current mode
            DCT_base = np.outer(dct_mat[mode_row], dct_mat[mode_col])

            # Obtain DCT residual R by convolution between image in spatial domain the current DCT base
            R = scipy.signal.convolve2d(x1, DCT_base, mode='valid')

            # Quantization, rounding, absolute value
            R = np.abs(tools.matlab.round(R / q))

            # Truncation
            R[R > T] = T

            # Iterate over the output histograms
            for hist_idx in np.arange(len(merged_coordinates)):
                sub_features = np.zeros(T + 1, dtype=int)

                for (row_shift, col_shift) in merged_coordinates[hist_idx]:
                    # Get quantized and truncated result at given shift
                    R_sub = R[row_shift::8, col_shift::8]

                    # Compute histogram and add up
                    sub_features += np.histogram(R_sub, bin_edges)[0]

                # Assign to output features array
                features[f'{mode_idx}_{hist_idx}'] = sub_features / np.sum(sub_features)
                # features[mode_idx, hist_idx, :] = sub_features / np.sum(sub_features)

    return features
