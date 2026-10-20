"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np


def cooccurrence(columns: np.ndarray, T: int, absolute: bool = False) -> np.ndarray:
    """Co-occurrence matrix of the columns, without normalization.

    :param columns: values of shape [num_samples, order]
    :param T: truncation threshold
    :param absolute: values are non-negative, truncated to [0, T], otherwise to [-T, T]
    :return: co-occurrence matrix with T + 1, or 2T + 1, bins per dimension
    """
    low = 0 if absolute else -T
    columns = np.clip(columns, low, T)
    # bins centered at the integers
    bin_edges = np.arange(low, T + 2) - .5
    cooc, _ = np.histogramdd(columns, bins=[bin_edges] * columns.shape[1])
    return cooc


def stack_columns(A: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Stacks the coefficients of the target DCT modes from all blocks as columns.

    :param A: 2D DCT plane, of shape [height, width]
    :param targets: DCT mode coordinates (y, x), of shape [num_targets, 2],
        coordinates beyond 8 refer to the neighboring blocks
    :return: array of shape [num_blocks, num_targets]
    """
    height, width = A.shape
    # the targets span this many blocks vertically and horizontally
    span_y, span_x = targets.max(axis=0) // 8 + 1
    num_blocks_y = height // 8 + 1 - span_y
    num_blocks_x = width // 8 + 1 - span_x
    return np.stack([
        A[y:y + 8 * num_blocks_y:8, x:x + 8 * num_blocks_x:8].flatten(order='F')
        for y, x in targets
    ], axis=1).astype(int)


def stack_columns_integral(A: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Stacks all pairs of coefficients at offset (dy, dx) as two columns.

    Pairs with a DC coefficient are omitted.
    For intra-block offsets (|dy|, |dx| < 8), pairs across block boundaries are omitted.

    :param A: 2D DCT plane, of shape [height, width]
    :param dy: vertical offset
    :param dx: horizontal offset
    :return: array of shape [num_pairs, 2]
    """
    height, width = A.shape
    valid1 = np.ones(A.shape, dtype=bool)
    valid1[::8, ::8] = False  # DC
    valid2 = valid1.copy()

    # intra-block pairs must not cross the block boundary
    if max(abs(dy), abs(dx)) < 8:
        y1 = np.arange(height)
        y2 = y1 + dy
        crossing = (0 <= y2) & (y2 < height) & (y1 // 8 != y2 // 8)
        valid1[y1[crossing], :] = False
        valid2[y2[crossing], :] = False
        x1 = np.arange(width)
        x2 = x1 + dx
        crossing = (0 <= x2) & (x2 < width) & (x1 // 8 != x2 // 8)
        valid1[:, x1[crossing]] = False
        valid2[:, x2[crossing]] = False

    # first and second element of each pair
    rows1, rows2 = slice(max(0, -dy), height - max(0, dy)), slice(max(0, dy), height + min(0, dy))
    cols1, cols2 = slice(max(0, -dx), width - max(0, dx)), slice(max(0, dx), width + min(0, dx))
    columns = np.stack([
        A[rows1, cols1].flatten(order='F'),
        A[rows2, cols2].flatten(order='F'),
    ], axis=1)
    valid = valid1[rows1, cols1].flatten(order='F') & valid2[rows2, cols2].flatten(order='F')
    return columns[valid]
