"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np


def normalize(f):
    """
    Normalize given feature vector to unit sum
    :param f: feature vector, arbitrary shape
    :return: normalized feature vector
    """
    f_sum = np.sum(f)
    if f_sum == 0:
        return f

    return f / f_sum


def sign_symmetrize(f):
    """
    Merge sign-symmetric entries, i.e., (-2, -2) == (+2, +2), (-1, -2) == (+1, +2), ...
    :param f: multi-dimensional co-occurrence matrix
    :return: 1D ndarray, symmetrized co-occurrence matrix flattened in column-major order with duplicate cells removed
    """

    size = f.shape[0]
    T = (size - 1) // 2

    if 2 == T:
        # Co-occurrence matrix (with cells named in Fortran order)
        #    | -2 -1  0 +1 +2
        # ---|----------------
        # -2 |  0  5 10 15 20
        # -1 |  1  6 11 16 21
        #  0 |  2  7 12 17 22
        #  1 |  3  8 13 18 23
        #  2 |  4  9 14 19 24

        # Merge sign-symmetric entries, i.e., (-2, -2) == (+2, +2), (-1, -2) == (+1, +2), ...
        bins_left = [0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7]
        bins_right = [24, 19, 14, 9, 4, 23, 18, 13, 8, 3, 22, 17]

    elif 5 == T:
        bins_left = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 1, 12, 23, 34, 45, 56, 67, 78, 89, 100, 111, 2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 3, 14, 25, 36, 47, 58, 69, 80, 91, 102, 113, 4, 15, 26, 37, 48, 59, 70, 81, 92, 103, 114, 5, 16, 27, 38, 49]
        bins_right = [120, 109, 98, 87, 76, 65, 54, 43, 32, 21, 10, 119, 108, 97, 86, 75, 64, 53, 42, 31, 20, 9, 118, 107, 96, 85, 74, 63, 52, 41, 30, 19, 8, 117, 106, 95, 84, 73, 62, 51, 40, 29, 18, 7, 116, 105, 94, 83, 72, 61, 50, 39, 28, 17, 6, 115, 104, 93, 82, 71]

    else:

        raise NotImplementedError()

    # Convert to float, otherwise the division by two may be list in the integer array
    f = f.flatten(order="F").astype(float)
    f[bins_left] = (f[bins_left] + f[bins_right]) / 2
    f = np.delete(f, bins_right)

    return f
