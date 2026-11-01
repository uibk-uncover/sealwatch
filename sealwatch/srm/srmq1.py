
from collections import OrderedDict
import numpy as np
from pathlib import Path
from typing import Union, Dict

from . import srm


def extract_from_file(
    path: Union[str, Path],
    **kw,
) -> Dict[str, np.ndarray]:
    """Extract SRMQ1 features from a grayscale image file, decoded with PIL.

    :param path: a grayscale image, e.g., PNG
    :type path: str or pathlib.Path
    :param kw: other parameters of extract()
    :return: SRMQ1 features, see extract()
    :rtype: collections.OrderedDict

    :Example:

    >>> features = sw.srmq1.extract_from_file('seal1.png')
    """
    return srm.extract_from_file(path=path, **kw, qs=[[1]]*5)


def extract(
    x: np.ndarray,
    **kw,
) -> Dict[str, np.ndarray]:
    """Extracts spatial rich model for steganalysis.

    :param x: grayscale image of shape [height, width], in [0, 255]
    :type x: np.ndarray
    :return: structured SRMQ1 features
    :rtype: collections.OrderedDict

    The Rust implementation is used by default, see :doc:`/guide/backends`.

    :Example:

    >>> features = sw.srmq1.extract(x)
    """
    return srm.extract(x, **kw, qs=[[1]]*5)
