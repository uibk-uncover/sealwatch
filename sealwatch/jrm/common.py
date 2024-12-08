"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np


# Reference 2D mode index using mode ids
mode_id_to_idx_map = {
    0: (0, 0),
    1: (0, 1),
    2: (0, 2),
    3: (0, 3),
    4: (0, 4),
    5: (0, 5),
    6: (1, 0),
    7: (1, 1),
    8: (1, 2),
    9: (1, 3),
    10: (1, 4),
    11: (2, 0),
    12: (2, 1),
    13: (2, 2),
    14: (2, 3),
    15: (3, 0),
    16: (3, 1),
    17: (3, 2),
    18: (4, 0),
    19: (4, 1),
    20: (5, 0),
    21: (1, 5),
    22: (2, 4),
    23: (3, 3),
    24: (4, 2),
    25: (5, 1),
    26: (2, 5),
    27: (3, 4),
    28: (4, 3),
    29: (5, 2),
    30: (3, 5),
    31: (4, 4),
    32: (5, 3),
    33: (4, 5),
    34: (5, 4),
    35: (5, 5)
}


def direction_to_coordinates(direction, a, b):
    """
    Translates a string direction into coordinates
    :param direction: string
    :param a: y-coordinate of current DCT mode
    :param b: x-coordinate of current DCT mode
    :return: ndarray of shape [2, 2]. The two rows correspond to the DCT modes over which to compute the co-occurrences. The two columns are the DCT modes' y- and x-coordinates, respectively.
    """
    if "sym_8x8" == direction:
        target = np.array([
            [a, b],
            [b, a],
        ])

    elif "inter_semidiag" == direction:
        target = np.array([
            [a + 8, b],
            [a, b + 8],
        ])

    elif "inter_symm" == direction:
        target = np.array([
            [a, b],
            [b, a + 8],
        ])

    elif "inter_hor" == direction:
        target = np.array([
            [a, b],
            [a, b + 8],
        ])

    elif "inter_diag" == direction:
        target = np.array([
            [a, b],
            [a + 8, b + 8],
        ])

    elif "hor" == direction:
        target = np.array([
            [a, b],
            [a, b + 1],
        ])

    elif "ver" == direction:
        target = np.array([
            [a, b],
            [a + 1, b],
        ])

    elif "diag" == direction:
        target = np.array([
            [a, b],
            [a + 1, b + 1],
        ])

    elif "semidiag" == direction:
        target = np.array([
            [a, b],
            [a - 1, b + 1],
        ])

    elif "hor_skip" == direction:
        target = np.array([
            [a, b],
            [a, b + 2],
        ])

    elif "diag_skip" == direction:
        target = np.array([
            [a, b],
            [a + 2, b + 2],
        ])

    elif "semidiag_skip" == direction:
        target = np.array([
            [a, b],
            [a - 2, b + 2],
        ])

    elif "horse" == direction:
        # Chess knight move
        target = np.array([
            [a, b],
            [a - 1, b + 2],
        ])

    else:
        raise ValueError("Unknown type")

    return target
