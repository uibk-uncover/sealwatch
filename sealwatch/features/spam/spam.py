"""
Implementation of the spam686 features as described in

T. Pevny, P. Bas and J. Fridrich
"Steganalysis by Subtractive Pixel Adjacency Matrix"
IEEE Transactions on Information Forensics and Security
Vol. 5, No. 2, pp. 215-224, June 2010
https://doi.org/10.1109/TIFS.2010.2045842

Author: Martin Benes
Affiliation: University of Innsbruck

This implementation builds on the original Matlab implementation provided by the paper authors. Please find the license of the original implementation below.
-------------------------------------------------------------------------
Copyright (c) 2011 DDE Lab, Binghamton University, NY. All Rights Reserved.
Permission to use, copy, modify, and distribute this software for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that this copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from DDE Lab. DDE Lab does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall Binghamton University or DDE Lab be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software. DDE Lab disclaims any warranties, and has no obligations to provide maintenance, support, updates, enhancements or modifications.
-------------------------------------------------------------------------
"""  # noqa: E501


import numpy as np
from collections import OrderedDict
from sealwatch.utils.jpeg import decompress_luminance_from_file
from sealwatch.utils.matlab import matlab_round
import typing


def extract_spam686_features_from_file(filepath: str) -> typing.Dict:
    """
    Extract SPAM features from luminance channel of given JPEG image
    :param filepath: JPEG image to be analzed
    :return: ordered dict with the feature values
    """
    luminance = decompress_luminance_from_file(filepath)

    return extract_spam686_features_from_img(luminance, T=3)


def extract_spam686_features_from_img(img, T=3, rounded: bool = True):
    """
    Extract 2nd-order spatial adjacency model (SPAM) features.
    The implementation merges over image directions.
    :param img: 2D ndarray
    :param T: truncation threshold
    :param rounded: Whether to round before the coocurrence.
    :return: ordered dict containing 686 feature dimensions in total.
    """
    features = OrderedDict()

    # horizontal left-right
    # Note that the naming is misleading. This actually computes the differences from right to left. But at the end, the left-right and right-left features are averaged anyway.
    D = img[:, :-1] - img[:, 1:]
    # Left
    L = D[:, 2:]
    # Center
    C = D[:, 1:-1]
    # Right
    R = D[:, :-2]
    Mh1 = get_m3(L, C, R, T, rounded=rounded)

    # Horizontal right-left
    D = -D
    L = D[:, :-2]
    C = D[:, 1:-1]
    R = D[:, 2:]
    Mh2 = get_m3(L, C, R, T, rounded=rounded)

    # Vertical bottom top
    D = img[:-1, :] - img[1:, :]
    L = D[2:, :]
    C = D[1:-1, :]
    R = D[:-2, :]
    Mv1 = get_m3(L, C, R, T, rounded=rounded)

    # Vertical top bottom
    D = -D
    L = D[:-2, :]
    C = D[1:-1, :]
    R = D[2:, :]
    Mv2 = get_m3(L, C, R, T, rounded=rounded)

    # diagonal left - right
    D = img[:-1, :-1] - img[1:, 1:]
    L = D[2:, 2:]
    C = D[1:-1, 1:- 1]
    R = D[:-2, :-2]
    Md1 = get_m3(L, C, R, T, rounded=rounded)

    # diagonal right - left
    D = -D
    L = D[:-2, :-2]
    C = D[1:-1, 1:-1]
    R = D[2:, 2:]
    Md2 = get_m3(L, C, R, T, rounded=rounded)

    # minor diagonal left - right
    D = img[1:, :-1] - img[:-1, 1:]
    L = D[:-2, 2:]
    C = D[1:-1, 1:-1]
    R = D[2:, :-2]
    Mm1 = get_m3(L, C, R, T, rounded=rounded)

    # minor diagonal right - left
    D = -D
    L = D[2:, :-2]
    C = D[1:-1, 1:-1]
    R = D[:-2, 2:]
    Mm2 = get_m3(L, C, R, T, rounded=rounded)

    # Average horizontal left - right, horizontal right - left, vertical bottom - top and vertical top - bottom
    features["straight"] = (Mh1 + Mh2 + Mv1 + Mv2) / 4

    # Average diagonals
    features["diagonal"] = (Md1 + Md2 + Mm1 + Mm2) / 4

    return features


def get_m3(L, C, R, T, rounded=False):
    """
    Calculate 3-D co-occurrences
    :param L: matrix with left pixel values
    :param C: matrix with center pixel values
    :param R: matrix with right pixel values
    :param T: truncation threshold
    :return: 3-D co-occurrence matrix of shape, where each dimension has 2 * T + 1 entries.
    """
    # Marginalization into borders
    L = np.clip(L.flatten(order="F"), -T, T)
    C = np.clip(C.flatten(order="F"), -T, T)
    R = np.clip(R.flatten(order="F"), -T, T)

    # # Round to integers
    if rounded:
        L = matlab_round(L)
        C = matlab_round(C)
        R = matlab_round(R)

    # Compute 3-D co-occurrences [-T, ..., +T]
    # np.histogramdd seems to be slower
    M = np.zeros((2 * T + 1, 2 * T + 1, 2 * T + 1), dtype=int)
    for i in range(-T, T + 1):
        ind = L == i
        C2 = C[ind]
        R2 = R[ind]
        for j in range(-T, T + 1):
            R3 = R2[C2 == j]
            for k in range(-T, T + 1):
                M[i + T, j + T, k + T] = np.sum(R3 == k)

    # Flattening and normalization
    M = M.flatten(order="F") / np.sum(M)

    return M
