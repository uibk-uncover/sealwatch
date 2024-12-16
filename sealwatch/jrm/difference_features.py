"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

from collections import OrderedDict

from .common import mode_id_to_idx_map, direction_to_coordinates
from .cooccurrence import extract_cooccurrences_from_columns, stack_cooccurrence_columns
from .symmetrization import sign_symmetrize, normalize

from .. import tools
# from sealwatch.utils.dict import append_features


def compute_difference_features(X1, X2, T=2):
    """
    Compute DCT-mode specific co-occurrences of differences of absolute values (horizontal/vertical)
    See Section 2.2 of the paper.
    Notation: C(x, y, dx, dy) denotes the co-occurrence between the two DCT modes (x, y) and (x + dx, y + dy).

    The features are calculated from X1, and with transposed indices from X2. The two parts are then averaged.

    :param X1: difference between some absolute-valued DCT coefficients, as 2D DCT plane. The first set of co-occurrences is calculated from this DCT plane.
    :param X2: difference between some other absolute-valued DCT coefficients, as 2D DCT plane. Calculate co-occurrences in transposed direction from this DCT plane before averaging the two co-occurrences.
    :param T: truncation threshold
    :return: ordered dict with feature groups
    """

    features = OrderedDict()

    # The next six groups capture intra-block relationships (1235 features in total)
    # G_h: Capture intra-block relationships in horizontal and vertical direction (after symmetrization) (260 features)
    # { C(x, y, 0, 1) | 0 <= x; 0 <= y; x + y <= 5 }
    # Note that (x=0, y=0) is omitted.
    mode_ids = list(range(1, 21))
    # There are 20 modes and the 2D co-occurrence matrix has shape (T + 1) ** 2, resulting in 20 * 16 = 320 feature dimensions.
    Dh = extract_submodels(X1, X2, mode_ids, T=T, direction="hor")
    tools.features.append(features, Dh, prefix=f"Dh_T{T}")

    # G_d: Union of diagonally and minor-diagonally neighboring pairs (260 features)
    # { C(x, y, 1, 1) | 0 <= x <= y; x + y <= 5}
    mode_ids = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]
    Dd_diag = extract_submodels(X1, X2, mode_ids, T=T, direction="diag")
    tools.features.append(features, Dd_diag, prefix=f"Dd_T{T}_diag")

    # { C(x, y, 1, -1) | 0 <= x <= y; x + y <= 5}
    mode_ids = [6, 7, 8, 9, 10, 12, 13, 14, 17]
    Dd_semidiag = extract_submodels(X1, X2, mode_ids, T=T, direction="semidiag")
    tools.features.append(features, Dd_semidiag, prefix=f"Dd_T{T}_semidiag")

    # G_oh: "Skip one" horizontally neighboring pairs (182 features)
    # { C(x, y, 0, 2) | 0 <= x; 0 <= y; x + y <= 4}
    mode_ids = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 16, 18]
    Doh = extract_submodels(X1, X2, mode_ids, T=T, direction="hor_skip")
    tools.features.append(features, Doh, prefix=f"Doh_T{T}")

    # G_x: pairs symmetrically positioned w.r.t. the 8x8 block diagonal (117 features)
    # { C(x, y, y - x, x - y) | 0 <= x < y; x + y <= 5}
    mode_ids = [1, 2, 3, 4, 5, 8, 9, 10, 14]
    Dx = extract_submodels(X1, X2, mode_ids, T=T, direction="sym_8x8")
    tools.features.append(features, Dx, prefix=f"Dx_T{T}")

    # G_od: "Skip one" diagonal and minor diagonal pairs (221 features)
    # { C(x, y, 2, 2) | 0 <= x <= y; x + y <= 4}
    mode_ids = [1, 2, 3, 4, 7, 8, 9, 13]
    Dod_diag_skip = extract_submodels(X1, X2, mode_ids, T=T, direction="diag_skip")
    tools.features.append(features, Dod_diag_skip, prefix=f"Dod_T{T}_diag_skip")

    # { C(x, y, 2, -2) | 0 <= x < y; x + y <= 5}
    mode_ids = [11, 12, 13, 14, 22, 16, 17, 23, 24]
    Dod_semidiag_skip = extract_submodels(X1, X2, mode_ids, T=T, direction="semidiag_skip")
    tools.features.append(features, Dod_semidiag_skip, prefix=f"Dod_T{T}_semidiag_skip")

    # G_km: "Knight move" positioned pairs (195 features)
    # { C(x, y, -1, 2) | 1 <= x; 0 <= y; x + y <= 5}
    mode_ids = list(range(6, 21))
    Dm = extract_submodels(X1, X2, mode_ids, T=T, direction="horse")
    tools.features.append(features, Dm, prefix=f"Dm_T{T}")

    # The next four blocks capture inter-block relationships between coefficients from neighboring blocks (806 features in total)
    # G_ih: horizontal neighbors in the same DCT mode (260 features)
    # { C(x, y, 0, 8) | 0 <= x; 0 <= y; x + y <= 5 }
    mode_ids = list(range(1, 21))
    Dih = extract_submodels(X1, X2, mode_ids, T=T, direction="inter_hor")
    tools.features.append(features, Dih, prefix=f"Dih_T{T}")

    # G_id: Diagonal neighbors in the same mode (143 features)
    # { C(x, y, 8, 8) | 0 <= x <= y; x + y <= 5}
    mode_ids = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]
    Did = extract_submodels(X1, X2, mode_ids, T=T, direction="inter_diag")
    tools.features.append(features, Did, prefix=f"Did_T{T}")

    # G_im: Minor-diagonal neighbors in the same mode (143 features)
    # { C(x, y, -8, 8) | 0 <= x <= y; x + y <= 5}
    mode_ids = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]
    Dis = extract_submodels(X1, X2, mode_ids, T=T, direction="inter_semidiag")
    tools.features.append(features, Dis, prefix=f"Dis_T{T}")

    # G_ix: Horizontal neighbors in mode symmetrically positioned w.r.t. the 8x8 block diagonal (260 features)
    # { C(x, y, y - x, x - y + 8) | 0 <= x; 0 <= y; x + y <= 5}
    mode_ids = list(range(1, 21))
    Dix = extract_submodels(X1, X2, mode_ids, T=T, direction="inter_symm")
    tools.features.append(features, Dix, prefix=f"Dix_T{T}")

    return features


def extract_submodels(X1, X2, mode_ids, T, direction):
    """
    Extract multiple submodels with the same direction from the given list of DCT modes.
    Calculate co-occurrence from X1, calculate co-occurrence with flipped DCT coordinates from X2, then add up the two co-occurrences.

    :param X1: difference between some absolute-valued DCT coefficients, as 2D DCT plane. The first set of co-occurrences is calculated from this DCT plane.
    :param X2: difference between some other absolute-valued DCT coefficients, as 2D DCT plane. Calculate co-occurrences in transposed direction from this DCT plane before averaging the two co-occurrences.
    :param mode_ids: list of DCT modes to consider
    :param T: truncation threshold
    :param direction: direction of co-occurrences
    :return: ordered dict
    """

    features_buffer = OrderedDict()
    for mode_id in mode_ids:
        mode_y, mode_x = mode_id_to_idx_map[mode_id]
        features_submodel = extract_submodel(X1, X2, mode_y, mode_x, T, direction)
        features_buffer[f"mode_y_{mode_y}_x_{mode_x}"] = features_submodel

    return features_buffer


def extract_submodel(X1, X2, a, b, T, direction):
    """
    Calculate co-occurrence from X1, calculate co-occurrence with flipped DCT coordinates from X2, then add up the two co-occurrences.

    :param X1: difference between some absolute-valued DCT coefficients, as 2D DCT plane. The first set of co-occurrences is calculated from this DCT plane.
    :param X2: difference between some other absolute-valued DCT coefficients, as 2D DCT plane. Calculate co-occurrences in transposed direction from this DCT plane before averaging the two co-occurrences.
    :param a: vertical offset of DCT mode
    :param b: horizontal offset of DCT mode
    :param T: truncation threshold
    :param direction: direction of co-occurrence
    :return: features as 1D ndarray
    """
    # Translate the direction to a pair of DCT modes
    dct_modes_coordinates = direction_to_coordinates(direction, a, b)

    # Extract the co-occurrence features
    blocks1 = stack_cooccurrence_columns(X1, dct_modes_coordinates)
    f1 = extract_cooccurrences_from_columns(blocks1, T)

    # Extract 8x8 diagonally symmetric co-occurrence features
    blocks2 = stack_cooccurrence_columns(X2, dct_modes_coordinates[:, ::-1])
    f2 = extract_cooccurrences_from_columns(blocks2, T)

    # Normalize
    f1 = normalize(f1)
    f2 = normalize(f2)

    # Sign symmetrization
    f = normalize(sign_symmetrize(f1 + f2))
    return f
