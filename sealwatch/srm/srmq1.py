
from collections import OrderedDict
import numpy as np
from pathlib import Path
from typing import Union, Dict

from . import srm


def extract_from_file(
    path: Union[str, Path],
    **kw,
) -> Dict[str, np.ndarray]:
    return srm.extract_from_file(path=path, **kw, qs=[[1]]*5)


def extract(
    x: np.ndarray,
    **kw,
) -> Dict[str, np.ndarray]:
    """Extracts spatial rich model for steganalysis.

    :param x: 2D input image
    :type x: np.ndarray
    :return: structured SRMQ1 features
    :rtype: collections.OrderedDict
    """
    return srm.extract(x, **kw, qs=[[1]]*5)
