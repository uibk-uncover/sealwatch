"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np


def extract_cooccurrences_from_columns_abs_naive(columns, T):
    """
    Marginalize to [0, ..., T].
    No normalization involved.

    :param columns: columns of values from which we want to extract the co-occurrences. Assumes that these values are all non-negative.
    :param T: truncation threshold
    :return: 2D co-occurrence matrix of shape [T + 1, T + 1]
    """

    order = columns.shape[1]
    assert order == 2, "Expected 2D co-occurrences"

    # Clip values to [0, ..., T]
    columns[columns > T] = T

    # Possible values are [0, ..., T]
    value_candidates = np.arange(0, T + 1)

    # 2nd order co-occurrences
    cooc_mat = np.zeros((T + 1, T + 1), dtype=int)

    for i, i_val in enumerate(value_candidates):
        # Value at position 0 matches i
        # blocks_i contains only the remaining columns
        blocks_i = columns[columns[:, 0] == i_val, 1]
        if len(blocks_i) == 0:
            continue

        for j, j_val in enumerate(value_candidates):
            # Value at position 1 matches j
            cooc_mat[i, j] = np.sum(blocks_i == j_val)

    return cooc_mat


def extract_cooccurrences_from_columns_abs(columns, T):
    """
    Marginalize to [0, ..., T].
    No normalization involved.

    :param columns: columns of values from which we want to extract the co-occurrences. Assumes that these values are all non-negative.
    :param T: truncation threshold
    :return: 2D co-occurrence matrix of shape [T + 1, T + 1]
    """

    order = columns.shape[1]

    # Clip values to [0, ..., T]
    columns[columns > T] = T

    # The bins correspond to values [0, ..., T]
    # Bin 0: [-0.5, +0.5]
    # ...
    # Bin T: [T-0.5, T+0.5]
    bin_edges_1d = np.arange(0, T + 2) - 0.5
    bin_edges_nd = [bin_edges_1d for _ in range(order)]
    cooc_mat, _ = np.histogramdd(columns, bins=bin_edges_nd)

    return cooc_mat


def extract_cooccurrences_from_columns_naive(columns, T):
    """
    Marginalize to [-T, +T].
    No normalization involved.

    :param columns: columns of values from which we want to extract the co-occurrences. Values can be positive, negative, or zero.
    :param T: truncation threshold
    :return: nD co-occurrence table, n-dimensional, where each dimension has (2 * T + 1) values
    """

    # Co-occurrence order
    order = columns.shape[1]

    # Clip values beyond the range
    columns[columns < -T] = -T
    columns[columns > T] = T

    # Possible values are [-T, ..., 0, ..., T]
    candidates = list(range(-T, T + 1))
    num_candidates = 2 * T + 1

    if 1 == order:
        # 1st order co-occurrence (histogram)
        cooc_mat = np.zeros(num_candidates, dtype=int)
        for i, i_val in enumerate(candidates):
            cooc_mat[i] = np.sum(columns == i_val)

    elif 2 == order:
        # 2nd order co-occurrence
        cooc_mat = np.zeros((num_candidates, num_candidates), dtype=int)
        for i, i_val in enumerate(candidates):
            blocks_i = columns[columns[:, 0] == i_val, 1]
            if len(blocks_i) == 0:
                continue

            for j, j_val in enumerate(candidates):
                cooc_mat[i, j] = np.sum(blocks_i == j_val)

    elif 3 == order:
        # 3rd order co-occurrences
        cooc_mat = np.zeros((num_candidates, num_candidates, num_candidates), dtype=int)
        for i, i_val in enumerate(candidates):
            blocks_i = columns[columns[:, 0] == i_val, 1:]
            if len(blocks_i) == 0:
                continue

            for j, j_val in enumerate(candidates):
                blocks_ij = blocks_i[blocks_i[:, 0] == j_val, 1:]
                if len(blocks_ij) == 0:
                    continue

                for k, k_val in enumerate(candidates):
                    cooc_mat[i, j, k] = np.sum(blocks_ij == k_val)

    elif 4 == order:
        # 4th order co-occurrences
        cooc_mat = np.zeros((num_candidates, num_candidates, num_candidates, num_candidates), dtype=int)

        for i, i_val in enumerate(candidates):
            blocks_i = columns[columns[:, 0] == i_val, 1:]
            if len(blocks_i) == 0:
                continue

            for j, j_val in enumerate(candidates):
                blocks_ij = blocks_i[blocks_i[:, 0] == j_val, 1:]
                if len(blocks_ij) == 0:
                    continue

                for k, k_val in enumerate(candidates):
                    blocks_ijk = blocks_ij[blocks_ij[:, 0] == k_val, 1:]
                    if len(blocks_ijk) == 0:
                        continue

                    for l, l_val in enumerate(candidates):
                        cooc_mat[i, j, k, l] = np.sum(blocks_ijk == l_val)

    else:
        raise NotImplementedError("Co-occurrence calculation only implemented for orders 1, 2, 3, and 4")

    return cooc_mat


def extract_cooccurrences_from_columns(columns, T):
    """
    Marginalize to [-T, +T].
    No normalization involved.

    :param columns: columns of values from which we want to extract the co-occurrences. Values can be positive, negative, or zero.
    :param T: truncation threshold
    :return: nD co-occurrence table, n-dimensional, where each dimension has (2 * T + 1) values
    """

    # Co-occurrence order
    order = columns.shape[1]

    # Clip values beyond the range
    columns[columns < -T] = -T
    columns[columns > T] = T

    # The bins correspond to values [-T, ... 0, , ..., T]
    # np.histogram requires bin edges
    # Bin 0: [-T-0.5, -T+0.5]
    # ...
    # Bin 2T + 1: [T-0.5, T+0.5]
    bin_edges_1d = np.arange(-T, T + 2) - 0.5
    bin_edges_nd = [bin_edges_1d for _ in range(order)]
    cooc_mat, _ = np.histogramdd(columns, bins=bin_edges_nd)

    return cooc_mat


def stack_cooccurrence_columns(A, targets):
    """
    Take the target DCT modes and extracts their corresponding coefficients from the DCT plane A. Store them as individual columns.
    :param A: 2D array of DCT coefficients, shape [height, width]
    :param targets: ndarray of shape [num_targets, 2]
    :return: ndarray of shape [num_block_shifts, num_targets]
    """

    height, width = A.shape

    mask = get_mask(targets)
    mask_height, mask_width = mask.shape

    num_vertical_block_shifts = height // 8 + 1 - mask_height // 8
    num_horizontal_block_shifts = width // 8 + 1 - mask_width // 8

    num_targets = len(targets)
    columns = None

    for i in range(num_targets):
        y, x = targets[i]
        C = A[y:y + 8 * num_vertical_block_shifts:8, x:x + 8 * num_horizontal_block_shifts:8]

        if columns is None:
            columns = np.zeros((C.size, num_targets), dtype=int)

        columns[:, i] = C.flatten(order="F")

    return columns


def stack_columns_integral_inter_block(A, dx, dy):
    """

    :param A: 2D ndarray, e.g., DCT plane of DCT coefficients or differences between DCT coefficients
    :param dx: horizontal step between the two DCT modes to calculate co-occurrences
    :param dy: vertical step between the two DCT modes to calculate co-occurrences
    :return: ndarray of shape [num_items, 2], where the second dimension can be used to calculate co-occurrences
    """
    # Set up a dummy value to mark cells that should be removed later
    CONST = np.sqrt(2)

    # Convert input plane to float so that we can assign the dummy value to its cells
    A1 = A.astype(float, copy=True)
    A2 = A.astype(float, copy=True)

    # Overwrite DC values with CONST
    A1[::8, ::8] = CONST
    A2[::8, ::8] = CONST

    height, width = A.shape

    # Drop rows from A1 and A2 for co-occurrence
    if dy > 0:
        # Drop bottom rows from A1
        A1 = np.delete(A1, range(height - dy, height), axis=0)
        # Drop top rows from A2
        A2 = np.delete(A2, range(dy), axis=0)
    elif dy < 0:
        # Drop bottom rows from A2
        A2 = np.delete(A2, range(height + dy, height), axis=0)
        # Drop top rows from A1
        A1 = np.delete(A1, range(-dy), axis=0)

    # Drop columns from A1 and A2
    if dx > 0:
        # Drop right columns from A1
        A1 = np.delete(A1, range(width - dx, width), axis=1)
        # Drop left columns from A2
        A2 = np.delete(A2, range(dx), axis=1)

    elif dx < 0:
        # Drop right columns from A2
        A2 = np.delete(A2, range(width + dx, width), axis=1)
        # Drop left columns from A1
        A1 = np.delete(A1, range(-dx), axis=1)

    # Concatenate the flattened arrays for calculating the co-occurrences
    columns = np.stack([A1.flatten(order="F"), A2.flatten(order="F")], axis=1)

    # Drop rows where at least one value was flagged
    columns = columns[~np.isclose(columns[:, 0], CONST) & ~np.isclose(columns[:, 1], CONST)]

    return columns


def stack_columns_integral_intra_block(A, dx, dy):
    # Set up a dummy value to mark cells that should be removed later
    CONST = np.sqrt(2)

    # Convert input plane to float so that we can assign the dummy value to its cells
    A1 = A.astype(float, copy=True)
    A2 = A.astype(float, copy=True)

    # Overwrite DC values with CONST
    A1[::8, ::8] = CONST
    A2[::8, ::8] = CONST

    height, width = A.shape

    for j in range(height):
        # Iterate over the rows
        y1 = j
        y2 = y1 + dy
        if 0 <= y1 < height and 0 <= y2 < height and np.floor(y1 / 8) != np.floor(y2 / 8):
            # Avoid comparison across block boundaries
            # Disable a complete row
            A1[y1, :] = CONST
            A2[y2, :] = CONST

    for i in range(width):
        # Iterate over the columns
        x1 = i
        x2 = x1 + dx
        if 0 <= x1 < width and 0 <= x2 < width and np.floor(x1 / 8) != np.floor(x2 / 8):
            # Avoid comparison across block boundaries
            # Disable a complete column
            A1[:, x1] = CONST
            A2[:, x2] = CONST

    # Drop rows from A1 and A2 for co-occurrence
    if dy > 0:
        # Drop bottom rows from A1
        A1 = np.delete(A1, range(height - dy, height), axis=0)
        # Drop top rows from A2
        A2 = np.delete(A2, range(dy), axis=0)
    elif dy < 0:
        # Drop bottom rows from A2
        A2 = np.delete(A2, range(height + dy, height), axis=0)
        # Drop top rows from A1
        A1 = np.delete(A1, range(-dy), axis=0)

    # Drop columns from A1 and A2
    if dx > 0:
        # Drop right columns from A1
        A1 = np.delete(A1, range(width - dx, width), axis=1)
        # Drop left columns from A2
        A2 = np.delete(A2, range(dx), axis=1)

    elif dx < 0:
        # Drop right columns from A2
        A2 = np.delete(A2, range(width + dx, width), axis=1)
        # Drop left columns from A1
        A1 = np.delete(A1, range(-dx), axis=1)

    # Concatenate the flattened arrays for calculating the co-occurrences
    columns = np.stack([A1.flatten(order="F"), A2.flatten(order="F")], axis=1)

    # Drop rows where at least one value was flagged
    columns = columns[~np.isclose(columns[:, 0], CONST) & ~np.isclose(columns[:, 1], CONST)]

    return columns


def get_mask(targets):
    """
    Transform a list of DCT models of interest into a binary mask.
    Only the DCT modes of interested are flagged as True.
    :param targets: 2D array of shape [num_samples, 2], where targets[:, 0] is the y-coordinate and targets[:, 1] is the x-coordiante.
    :return: binary mask; depending on the given targets, the mask can have a shape between [8, 8], [16, 16], [24, 24], or [32, 32]
    """

    mask_height = 8
    mask_width = 8

    # Adjust mask height
    if np.any(targets[:, 0] >= 8) and np.all(targets[:, 0] < 16):
        mask_height = 16

    elif np.any(targets[:, 0] >= 16) and np.all(targets[:, 0] < 24):
        mask_height = 24

    elif np.any(targets[:, 0] >= 24) and np.all(targets[:, 0] < 32):
        mask_height = 32

    # Adjust mask width
    if np.any(targets[:, 1] >= 8) and np.all(targets[:, 1] < 16):
        mask_width = 16

    elif np.any(targets[:, 1] >= 16) and np.all(targets[:, 1] < 24):
        mask_width = 24

    elif np.any(targets[:, 1] >= 24) and np.all(targets[:, 1] < 32):
        mask_width = 32

    mask = np.zeros((mask_height, mask_width), dtype=bool)
    mask[targets[:, 0], targets[:, 1]] = True
    return mask
