"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

from collections import OrderedDict

# from sealwatch.features.jrm.common import mode_id_to_idx_map, direction_to_coordinates
# from sealwatch.features.jrm.cooccurrence import extract_cooccurrences_from_columns_abs, stack_cooccurrence_columns
# from sealwatch.features.jrm.symmetrization import normalize
# from sealwatch.utils.dict import append_features

from .. import tools
from .common import mode_id_to_idx_map, direction_to_coordinates
from .cooccurrence import extract_cooccurrences_from_columns_abs, stack_cooccurrence_columns
from .symmetrization import normalize


def compute_absolute_value_features(abs_X, T=3):
    """
    Compute DCT-mode specific co-occurrences of absolute values.
    See Section 2.2 of the paper.
    Notation: C(x, y, dx, dy) denotes the co-occurrence between the two DCT modes (x, y) and (x + dx, y + dy).

    The feature naming convention is "A{direction}_T{T}", where A stands for absolute values and T is the truncation threshold.

    :param abs_X: absolute values of DCT coefficients in 2D jpegio format
    :param T: truncation threshold
    :return: ordered dict with feature groups
    """

    features = OrderedDict()

    # The next six groups capture intra-block relationships (1520 features in total)
    # G_h: Capture intra-block relationships in horizontal and vertical direction (after symmetrization) (320 features)
    # { C(x, y, 0, 1) | 0 <= x; 0 <= y; x + y <= 5 }
    # Note that (x=0, y=0) is omitted.
    mode_ids = list(range(1, 21))
    # There are 20 modes and the 2D co-occurrence matrix has shape (T + 1) ** 2, resulting in 20 * 16 = 320 feature dimensions.
    Ah = extract_submodels_abs(abs_X, mode_ids, T=T, direction="hor")
    tools.features.append(features, Ah, prefix=f"Ah_T{T}")

    # G_d: Union of diagonally and minor-diagonally neighboring pairs (320 features)
    # { C(x, y, 1, 1) | 0 <= x <= y; x + y <= 5}
    mode_ids = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]
    Ad_diag = extract_submodels_abs(abs_X, mode_ids, T=T, direction="diag")
    tools.features.append(features, Ad_diag, prefix=f"Ad_T{T}_diag")

    # { C(x, y, 1, -1) | 0 <= x <= y; x + y <= 5}
    mode_ids = [6, 7, 8, 9, 10, 12, 13, 14, 17]
    Ad_semidiag = extract_submodels_abs(abs_X, mode_ids, T=T, direction="semidiag")
    tools.features.append(features, Ad_semidiag, prefix=f"Ad_T{T}_semidiag")

    # G_oh: "Skip one" horizontally neighboring pairs (224 features)
    # { C(x, y, 0, 2) | 0 <= x; 0 <= y; x + y <= 4}
    mode_ids = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 16, 18]
    Aoh = extract_submodels_abs(abs_X, mode_ids, T=T, direction="hor_skip")
    tools.features.append(features, Aoh, prefix=f"Aoh_T{T}")

    # G_x: pairs symmetrically positioned w.r.t. the 8x8 block diagonal (114 features)
    # { C(x, y, y - x, x - y) | 0 <= x < y; x + y <= 5}
    mode_ids = [1, 2, 3, 4, 5, 8, 9, 10, 14]
    Ax = extract_submodels_abs(abs_X, mode_ids, T=T, direction="sym_8x8")
    tools.features.append(features, Ax, prefix=f"Ax_T{T}")

    # G_od: "Skip one" diagonal and minor diagonal pairs (272 features)
    # { C(x, y, 2, 2) | 0 <= x <= y; x + y <= 4}
    mode_ids = [1, 2, 3, 4, 7, 8, 9, 13]
    Aod_diag_skip = extract_submodels_abs(abs_X, mode_ids, T=T, direction="diag_skip")
    tools.features.append(features, Aod_diag_skip, prefix=f"Aod_T{T}_diag_skip")

    # { C(x, y, 2, -2) | 0 <= x < y; x + y <= 5}
    mode_ids = [11, 12, 13, 14, 22, 16, 17, 23, 24]
    Aod_semidiag_skip = extract_submodels_abs(abs_X, mode_ids, T=T, direction="semidiag_skip")
    tools.features.append(features, Aod_semidiag_skip, prefix=f"Aod_T{T}_semidiag_skip")

    # G_km: "Knight move" positioned pairs (240 features)
    # { C(x, y, -1, 2) | 1 <= x; 0 <= y; x + y <= 5}
    mode_ids = list(range(6, 21))
    Am = extract_submodels_abs(abs_X, mode_ids, T=T, direction="horse")
    tools.features.append(features, Am, prefix=f"Am_T{T}")

    # The next four blocks capture inter-block relationships between coefficients from neighboring blocks (992 features in total)
    # G_ih: horizontal neighbors in the same DCT mode (320 features)
    # { C(x, y, 0, 8) | 0 <= x; 0 <= y; x + y <= 5 }
    mode_ids = list(range(1, 21))
    Aih = extract_submodels_abs(abs_X, mode_ids, T=T, direction="inter_hor")
    tools.features.append(features, Aih, prefix=f"Aih_T{T}")

    # G_id: Diagonal neighbors in the same mode (176 features)
    # { C(x, y, 8, 8) | 0 <= x <= y; x + y <= 5}
    mode_ids = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]
    Aid = extract_submodels_abs(abs_X, mode_ids, T=T, direction="inter_diag")
    tools.features.append(features, Aid, prefix=f"Aid_T{T}")

    # G_im: Minor-diagonal neighbors in the same mode (176 features)
    # { C(x, y, -8, 8) | 0 <= x <= y; x + y <= 5}
    mode_ids = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]
    Ais = extract_submodels_abs(abs_X, mode_ids, T=T, direction="inter_semidiag")
    tools.features.append(features, Ais, prefix=f"Ais_T{T}")

    # G_ix: Horizontal neighbors in mode symmetrically positioned w.r.t. the 8x8 block diagonal (320 features)
    # { C(x, y, y - x, x - y + 8) | 0 <= x; 0 <= y; x + y <= 5}
    mode_ids = list(range(1, 21))
    Aix = extract_submodels_abs(abs_X, mode_ids, T=T, direction="inter_symm")
    tools.features.append(features, Aix, prefix=f"Aix_T{T}")

    return features


def extract_submodels_abs(X, mode_ids, T, direction):
    """
    Extract multiple submodels with the same direction from the given list of DCT modes
    :param X: absolute-valued DCT coefficients as 2D ndarray
    :param mode_ids: list of DCT modes to consider
    :param T: truncation threshold
    :param direction: direction of co-occurrences
    :return: ordered dict
    """
    features_buffer = OrderedDict()
    for mode_id in mode_ids:
        mode_y, mode_x = mode_id_to_idx_map[mode_id]
        features_submodel = extract_submodel_abs(X, mode_y, mode_x, T, direction)
        features_buffer[f"mode_y_{mode_y}_x_{mode_x}"] = features_submodel

    return features_buffer


def extract_submodel_abs(X, a, b, T, direction):
    """

    :param X: absolute values of DCT coefficients
    :param a: vertical index of DCT mode
    :param b: horizontal index of DCT mode
    :param T: truncation threshold
    :param direction: direction of co-occurrence
    :return: features as 1D ndarray
    """
    # Translate the direction to a pair of DCT modes
    dct_coordinates = direction_to_coordinates(direction, a, b)

    blocks1 = stack_cooccurrence_columns(X, dct_coordinates)
    f1 = extract_cooccurrences_from_columns_abs(blocks1, T)

    # Swap horizontal and vertical direction
    blocks2 = stack_cooccurrence_columns(X, dct_coordinates[:, ::-1])
    f2 = extract_cooccurrences_from_columns_abs(blocks2, T)

    # Normalize
    f1 = normalize(f1)
    f2 = normalize(f2)

    # Flatten
    f1 = f1.flatten(order="F")
    f2 = f2.flatten(order="F")

    # Average over the two directions
    return (f1 + f2) / 2
