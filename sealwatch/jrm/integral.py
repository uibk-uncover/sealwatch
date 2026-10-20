"""Integral features of JRM, co-occurrences summed over all DCT modes.

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

from collections import OrderedDict
import numpy as np
from typing import Dict

from .common import normalize, sign_symmetrize
from .cooccurrence import cooccurrence, stack_columns_integral


# co-occurrences of absolute values, summed over the (dy, dx) offsets
ABSOLUTE = {
    'MXh': [(0, 1), (1, 0)],  # horizontal and vertical
    'MXd': [(1, 1)],  # diagonal
    'MXs': [(-1, 1), (1, -1)],  # semi-diagonal, both directions
    'MXih': [(0, 8), (8, 0)],  # inter-block horizontal and vertical
    'MXid': [(8, -8), (8, 8)],  # inter-block diagonal and semi-diagonal
}

# co-occurrences of differences, by (dy, dx) offset
# frequency (intra-block) dependencies
FREQUENCY = {
    'MXh': (0, 1),
    'MXv': (1, 0),
    'Mxd': (1, 1),
    'Mxs': (-1, 1),
}
# spatial (inter-block) dependencies
SPATIAL = {
    'MXih': (0, 8),
    'MXiv': (8, 0),
    'MXid': (8, 8),
    'MXis': (-8, 8),
}


def compute_integral_features(
    abs_X: np.ndarray,
    differences: Dict[str, np.ndarray],
    T: int = 5,
) -> OrderedDict:
    """Integral features, capturing a large range of DCT coefficients across the DCT plane.

    The high number of DCT coefficients allows for a larger truncation threshold T.
    DC modes are omitted.

    :param abs_X: absolute values of DCT coefficients, as 2D DCT plane
    :param differences: differences of absolute values, by name H, V, D, IH, IV
    :param T: truncation threshold
    :return: ordered dict with the integral submodels, 2620 features in total
    """
    features = OrderedDict()

    # absolute values (5 x 36 = 180 features)
    for name, offsets in ABSOLUTE.items():
        f = sum(cooccurrence(stack_columns_integral(abs_X, dy, dx), T=T, absolute=True) for dy, dx in offsets)
        features[f'Ax_T{T}_{name}'] = normalize(f.flatten(order='F'))

    # differences (5 x 8 x 61 = 2440 features)
    for diff_name, X in differences.items():
        for prefix, offsets in [('Df', FREQUENCY), ('Ds', SPATIAL)]:
            for name, (dy, dx) in offsets.items():
                f = cooccurrence(stack_columns_integral(X, dy, dx), T=T)
                features[f'{prefix}{diff_name}_T{T}_{name}'] = normalize(sign_symmetrize(f))

    return features
