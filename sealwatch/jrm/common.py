"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np


# DCT modes (y, x) by mode id, first those with x + y <= 5
MODES = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 0), (3, 1), (3, 2),
    (4, 0), (4, 1),
    (5, 0),
    (1, 5), (2, 4), (3, 3), (4, 2), (5, 1),
    (2, 5), (3, 4), (4, 3), (5, 2),
    (3, 5), (4, 4), (5, 3),
    (4, 5), (5, 4),
    (5, 5),
]

# Pair of DCT coordinates for co-occurrence of mode (a, b), by direction
DIRECTIONS = {
    'hor': lambda a, b: [(a, b), (a, b + 1)],
    'diag': lambda a, b: [(a, b), (a + 1, b + 1)],
    'semidiag': lambda a, b: [(a, b), (a - 1, b + 1)],
    'hor_skip': lambda a, b: [(a, b), (a, b + 2)],
    'diag_skip': lambda a, b: [(a, b), (a + 2, b + 2)],
    'semidiag_skip': lambda a, b: [(a, b), (a - 2, b + 2)],
    'horse': lambda a, b: [(a, b), (a - 1, b + 2)],  # chess knight move
    'sym_8x8': lambda a, b: [(a, b), (b, a)],
    'inter_hor': lambda a, b: [(a, b), (a, b + 8)],
    'inter_diag': lambda a, b: [(a, b), (a + 8, b + 8)],
    'inter_semidiag': lambda a, b: [(a + 8, b), (a, b + 8)],
    'inter_symm': lambda a, b: [(a, b), (b, a + 8)],
}


def direction_to_coordinates(direction: str, a: int, b: int) -> np.ndarray:
    """Translates a direction into the coordinates of the two DCT modes.

    :param direction: direction of the co-occurrence, key of DIRECTIONS
    :param a: y-coordinate of the DCT mode
    :param b: x-coordinate of the DCT mode
    :return: array of shape [2, 2], rows are the DCT modes, columns their y- and x-coordinates
    """
    return np.array(DIRECTIONS[direction](a, b))


def normalize(f: np.ndarray) -> np.ndarray:
    """Normalizes the features to unit sum, unless all are zero."""
    f_sum = np.sum(f)
    if f_sum == 0:
        return f
    return f / f_sum


def sign_symmetrize(f: np.ndarray) -> np.ndarray:
    """Merges sign-symmetric cells of a 2D co-occurrence matrix, i.e., (-2, -1) with (+2, +1).

    :param f: co-occurrence matrix of shape [2T+1, 2T+1], indexed from -T to T
    :return: 1D array, column-major flattened, with the duplicate cells removed
    """
    n = f.shape[0]
    # first half of the cells in row-major order, and their point reflections
    k = np.arange(n * n // 2)
    bins_left = k // n + (k % n) * n
    bins_right = n * n - 1 - bins_left
    # float, as the merged cells are averaged
    f = f.flatten(order='F').astype(float)
    f[bins_left] = (f[bins_left] + f[bins_right]) / 2
    return np.delete(f, bins_right)
