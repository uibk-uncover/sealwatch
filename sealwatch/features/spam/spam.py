import numpy as np
from collections import OrderedDict
from sealwatch.utils.jpeg import decompress_luminance_from_filepath
from sealwatch.utils.matlab import matlab_round
import typing


def extract_spam686_features_from_filepath(filepath: str) -> typing.Dict:
    """
    Extract SPAM features from luminance channel of given JPEG image
    :param filepath: JPEG image to be analzed
    :return: ordered dict with the feature values
    """
    luminance = decompress_luminance_from_filepath(filepath)

    return extract_spam686_features(luminance, T=3)


def extract_spam686_features(img, T=3, rounded: bool = True):
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
