"""

Authors: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

from collections import OrderedDict
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union

from . import spam as py
from .._sealwatch import spam as rs
from .. import tools


def extract(
    x: np.ndarray,
    *,
    T: int = 3,
    rounded: bool = False,
) -> OrderedDict:
    """
    Extract 2nd-order spatial adjacency model (SPAM) features.
    The implementation merges over image directions.

    The final feature set has 686 dimensions.

    :param x: 2D ndarray
    :type x: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param T: truncation threshold
    :type T: int
    :return: ordered dict containing 686 feature dimensions in total.
    :rtype: collections.OrderedDict

    :Examples:

    >>> features = sw.spam.extract(x)

    By default, this function uses Rust-accelerated backend. To use the (substantially slower) Python implementation, type

    >>> with sw.BACKEND_PYTHON:
    >>>     features = sw.spam.extract(x)
    """
    # choose implementation
    backend = tools.get_backend()
    if backend == tools.BACKEND_RUST:
        # check types
        if x.dtype != np.uint8:
            raise TypeError('parameter x must be uint8')
        #
        features = rs.extract(x=x, T=T, rounded=rounded)

    elif backend == tools.BACKEND_PYTHON:
        # check types
        if x.dtype != np.uint8 and not rounded:
            raise TypeError('parameter x must be uint8 or rounding must be on')
        #
        features = py.extract(x=x, T=T, rounded=rounded)

    else:
        raise NotImplementedError(f'unknown backend {backend}')

    #
    return features


def extract_from_file(
    path: Union[str, Path],
    **kw,
) -> OrderedDict:
    """
    Extract SPAM features from a grayscale image file, decoded with PIL.

    :param path: grayscale image, e.g., PNG or JPEG
    :type path: str or pathlib.Path
    :param kw: other parameters of extract()
    :return: ordered dict with the feature values
    :rtype: collections.OrderedDict

    :Example:

    >>> features = sw.spam.extract_from_file('seal1.png')
    """
    x = np.array(Image.open(path))
    return extract(x, **kw)
