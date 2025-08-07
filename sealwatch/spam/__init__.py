"""

Authors: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

from collections import OrderedDict
import numpy as np
from pathlib import Path
from typing import Union
import warnings

from . import spam as py
from .._sealwatch import spam as rs
from .. import tools


def extract(
    x1: np.ndarray,
    *,
    T: int = 3,
    rounded: bool = False,
) -> OrderedDict:
    """
    Extract 2nd-order spatial adjacency model (SPAM) features.
    The implementation merges over image directions.

    The final feature set has 686 dimensions.

    :param x1: 2D ndarray
    :type x1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param T: truncation threshold
    :type T: int
    :return: ordered dict containing 686 feature dimensions in total.
    :rtype: collections.OrderedDict

    :Examples:

    >>> features = sw.spam.extract(x1)

    By default, this function uses Rust-accelerated backend. To use the (substantially slower) Python implementation, type

    >>> with sw.BACKEND_PYTHON:
    >>>     features = sw.spam.extract(x1)
    """
    # choose implementation
    backend = tools.get_backend()
    if backend == tools.BACKEND_RUST:
        # check types
        if x1.dtype != np.uint8:
            raise TypeError('parameter x1 must be uint8')
        #
        features = rs.extract(x1=x1, T=T, rounded=rounded)

    elif backend == tools.BACKEND_PYTHON:
        # check types
        if x1.dtype != np.uint8 and not rounded:
            raise TypeError('parameter x1 must be uint8 or rounding must be on')
        #
        features = py.extract(x1=x1, T=T, rounded=rounded)

    else:
        raise NotImplementedError(f'unknown backend {backend}')

    #
    return features


def extract_from_file(
    path: Union[str, Path],
    *,
    rounded: bool = True,
    **kw,
) -> OrderedDict:
    """
    Extract SPAM features from luminance channel of given JPEG image

    :param path: JPEG image to be analzed
    :type path: str or pathlib.Path
    :return: ordered dict with the feature values
    :rtype: collections.OrderedDict

    :Example:

    >>> # TODO

    This function can only work with Python backend.
    """
    backend = tools.get_backend()
    if backend != tools.BACKEND_PYTHON:
        warnings.warn(f'backend {backend} not supported for extract_from_file(), falling back to Python')
    #
    x1 = tools.jpeg.decompress_from_file(path)
    #
    with tools.BACKEND_PYTHON:
        return extract(x1=x1, **kw, rounded=rounded)
