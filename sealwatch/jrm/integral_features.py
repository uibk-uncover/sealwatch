"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

from collections import OrderedDict

from .cooccurrence import extract_cooccurrences_from_columns, extract_cooccurrences_from_columns_abs, stack_columns_integral_inter_block, stack_columns_integral_intra_block
from .symmetrization import sign_symmetrize, normalize


def compute_integral_features(abs_X, abs_X_Dh, abs_X_Dv, abs_X_Dd, abs_X_Dih, abs_X_Div, T=5):
    """
    Integral features capture a large range of DCT coefficients across the DCT plane. The high number of DCT coefficients allows for a larger truncation threshold T, e.g., T = 5, without too many unpopulated bins.
    The total number of integral features is 2620.

    :param abs_X: 2D ndarray, absolute-valued DCT coefficients
    :param abs_X_Dh: 2D ndarray, differences between absolute-valued DCT coefficients in horizontal direction
    :param abs_X_Dv: 2D ndarray, differences between absolute-valued DCT coefficients in vertical direction
    :param abs_X_Dd: 2D ndarray, differences between absolute-valued DCT coefficients in diagonal direction
    :param abs_X_Dih: 2D ndarray, inter-block differences between absolute-valued DCT coefficients in horizontal direction
    :param abs_X_Div: 2D ndarray, inter-block differences between absolute-valued DCT coefficients in vertical direction
    :param T: truncation threshold T
    :return: ordered dict with integral feature submodels
    """

    features = OrderedDict()

    # Integral co-occurrences from absolute values Ax (180 features)
    # Cover both the spatial (inter-block) and frequency (inter-block) dependencies.
    # I^x = { sum_{x, y} C(x, y, dx, dy) | (dx, dy) in {(0, 1), (1, 1), (1, -1), (0, 8), (8, 8)}}
    subfeatures = extract_specific_integral_features_abs(abs_X, T=T, direction="MXh") # Sum of horizontal and vertical direction
    features[f"Ax_T{T}_MXh"] = subfeatures

    subfeatures = extract_specific_integral_features_abs(abs_X, T=T, direction="MXd") # Diagonal direction
    features[f"Ax_T{T}_MXd"] = subfeatures

    subfeatures = extract_specific_integral_features_abs(abs_X, T=T, direction="MXs") # Semi-diagonal direction
    features[f"Ax_T{T}_MXs"] = subfeatures

    subfeatures = extract_specific_integral_features_abs(abs_X, T=T, direction="MXih") # Sum of inter-block horizontal direction and inter-block vertical direction
    features[f"Ax_T{T}_MXih"] = subfeatures

    subfeatures = extract_specific_integral_features_abs(abs_X, T=T, direction="MXid") # Inter-block diagonal direction
    features[f"Ax_T{T}_MXid"] = subfeatures

    # Difference-based submodels
    diff_name_coeffs_pairs = [
        ("H", abs_X_Dh), # horizontal differences
        ("V", abs_X_Dv), # vertical differences
        ("D", abs_X_Dd), # diagonal differences
        ("IH", abs_X_Dih), # inter-block horizontal differences
        ("IV", abs_X_Div), # inter-block vertical differences
    ]

    for name, X in diff_name_coeffs_pairs:
        # Capture frequency (intra-block) dependencies (5 * 4 * 61 = 5 * 244 = 1220 features)
        # I_f^* = { sum_{x, y} C(x, y, dx, dy) | (dx, dy) in {(0, 1), (1, 0), (1, 1), (1, -1)}}
        # DC modes are omitted in all definitions.
        # For intra-block pairs, the summation is always over all DCT modes (x, y) in {0, ..., 7}^2 such that both (x, y) and (x + dx, y + dy) lie within the same 8x8 block.
        # The same constraint applies to inter-block matrices whenever the indices would end up outside of the DCT array.
        subfeatures = extract_specific_integral_features_diff2(X, T=T, direction="MXh") # horizontal (0, 1)
        features[f"Df{name}_T{T}_MXh"] = subfeatures

        subfeatures = extract_specific_integral_features_diff2(X, T=T, direction="MXv") # vertical (1, 0)
        features[f"Df{name}_T{T}_MXv"] = subfeatures

        subfeatures = extract_specific_integral_features_diff2(X, T=T, direction="MXd") # diagonal (1, 1)
        features[f"Df{name}_T{T}_Mxd"] = subfeatures

        subfeatures = extract_specific_integral_features_diff2(X, T=T, direction="MXs") # semi-diagonal (1, -1)
        features[f"Df{name}_T{T}_Mxs"] = subfeatures

        # Capture spatial (inter-block) dependencies (5 * 4 * 61 features = 5 * 244 = 1220 features)
        # I_s^* = { sum_{x, y} C(x, y, dx, dy) | (dx, dy) in {(0, 8), (8, 0), (8, 8), (8, -8)}}
        subfeatures = extract_specific_integral_features_diff2(X, T=T, direction="MXih") # horizontal (0, 8)
        features[f"Ds{name}_T{T}_MXih"] = subfeatures

        subfeatures = extract_specific_integral_features_diff2(X, T=T, direction="MXiv") # vertical (8, 0)
        features[f"Ds{name}_T{T}_MXiv"] = subfeatures

        subfeatures = extract_specific_integral_features_diff2(X, T=T, direction="MXid")  # diagonal (8, 8)
        features[f"Ds{name}_T{T}_MXid"] = subfeatures

        subfeatures = extract_specific_integral_features_diff2(X, T=T, direction="MXis") # semi-diagonal (8, -8)
        features[f"Ds{name}_T{T}_MXis"] = subfeatures

    return features


def extract_specific_integral_features_abs(A, T, direction):
    if "MXh" == direction:
        # Average over horizontal left->right and vertical top->bottom

        # Horizontal left->right (0, 1)
        columns = stack_columns_integral_intra_block(A, dy=0, dx=1)
        features_left_right = extract_cooccurrences_from_columns_abs(columns, T=T)

        # Vertical top->bottom (1, 0)
        columns = stack_columns_integral_intra_block(A, dy=1, dx=0)
        features_top_bottom = extract_cooccurrences_from_columns_abs(columns, T=T)

        features = (features_left_right + features_top_bottom)
        features = normalize(features.flatten(order="F"))
        return features

    elif "MXd" == direction:
        # Diagonal from DC (1, 1)
        columns = stack_columns_integral_intra_block(A, dy=1, dx=1)
        features = extract_cooccurrences_from_columns_abs(columns, T=T)
        features = normalize(features.flatten(order="F"))
        return features

    elif "MXs" == direction:
        # Semi-diagonal both directions
        columns = stack_columns_integral_intra_block(A, dy=-1, dx=+1)
        features_top_right = extract_cooccurrences_from_columns_abs(columns, T=T)

        columns = stack_columns_integral_intra_block(A, dy=+1, dx=-1)
        features_bottom_left = extract_cooccurrences_from_columns_abs(columns, T=T)

        features = (features_top_right + features_bottom_left)
        features = normalize(features.flatten(order="F"))
        return features

    elif "MXih" == direction:
        # Inter horizontal/vertical
        columns = stack_columns_integral_inter_block(A, dy=0, dx=8)
        features_horizontal = extract_cooccurrences_from_columns_abs(columns, T=T)

        columns = stack_columns_integral_inter_block(A, dy=8, dx=0)
        features_vertical = extract_cooccurrences_from_columns_abs(columns, T=T)

        features = (features_horizontal + features_vertical)
        features = normalize(features.flatten(order="F"))
        return features

    elif "MXid" == direction:
        # Inter diagonal/semidiagonal
        columns = stack_columns_integral_inter_block(A, dy=8, dx=-8)
        features_bottom_left = extract_cooccurrences_from_columns_abs(columns, T=T)

        columns = stack_columns_integral_inter_block(A, dy=8, dx=8)
        features_top_right = extract_cooccurrences_from_columns_abs(columns, T=T)

        features = (features_bottom_left + features_top_right)
        features = normalize(features.flatten(order="F"))
        return features

    raise ValueError("Unknown direction")


def extract_specific_integral_features_diff2(X, T, direction):
    """
    Compute co-occurrence features in the given direction fromdifference between absolute-valued DCT coefficients
    :param X: 2D ndarray, difference between DCT coefficients
    :param T: truncation threshold
    :param direction: type of co-occurrences
    :return: features
    """

    if "MXh" == direction:
        # Intra-block horizontal
        dy = 0
        dx = 1
        inter_block = False
    elif "MXv" == direction:
        # Intra-block vertical
        dy = 1
        dx = 0
        inter_block = False
    elif "MXd" == direction:
        # Intra-block diagonal
        dy = 1
        dx = 1
        inter_block = False
    elif "MXs" == direction:
        # Intra-block semi-diagonal
        dy = -1
        dx = 1
        inter_block = False
    elif "MXih" == direction:
        # Inter-block horizontal
        dy = 0
        dx = 8
        inter_block = True
    elif "MXiv" == direction:
        # Inter-block vertical
        dy = 8
        dx = 0
        inter_block = True
    elif "MXid" == direction:
        # Inter-block diagonal
        dy = 8
        dx = 8
        inter_block = True
    elif "MXis" == direction:
        # Inter-block semi-diagonal
        dy = -8
        dx = 8
        inter_block = True
    else:
        raise ValueError("Unknown direction")

    if inter_block:
        columns = stack_columns_integral_inter_block(X, dy=dy, dx=dx)
    else:
        columns = stack_columns_integral_intra_block(X, dy=dy, dx=dx)

    features = extract_cooccurrences_from_columns(columns, T=T)
    features = sign_symmetrize(features)
    features = normalize(features)
    return features
