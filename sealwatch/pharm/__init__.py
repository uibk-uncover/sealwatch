"""

Authors: Benedikt Lorch
Affiliation: University of Innsbruck
"""

from collections import OrderedDict
import enum
import numpy as np
from pathlib import Path
from typing import Union

from .pharm_original import extract_original_from_file, extract_original
from .pharm_revisited import extract_revisited, extract_revisited_from_file, PharmRevisitedFeatureExtractor


class Implementation(enum.Enum):
    """PHARM implementation to choose from."""

    PHARM_ORIGINAL = enum.auto()
    """Original PHARM implementation by DDE."""
    PHARM_REVISITED = enum.auto()
    """PHARM implementation with fixes."""


def extract(
    x1: np.ndarray,
    *,
    implementation: Implementation = Implementation.PHARM_REVISITED,
    **kw,
) -> OrderedDict:
    #
    if implementation == Implementation.PHARM_REVISITED:
        return extract_original(
            x1,
            **kw,
        )
    #
    elif implementation == Implementation.PHARM_ORIGINAL:
        return extract_revisited(
            x1,
            **kw,
        )
    #
    else:
        raise NotImplementedError(f'unknown implementation {implementation}')


def extract_from_file(
    path: Union[str, Path],
    *,
    implementation: Implementation = Implementation.PHARM_REVISITED,
    **kw,
) -> OrderedDict:
    #
    if implementation == Implementation.PHARM_REVISITED:
        return extract_original_from_file(
            path,
            **kw,
        )
    #
    elif implementation == Implementation.PHARM_ORIGINAL:
        return extract_revisited_from_file(
            path,
            **kw,
        )
    #
    else:
        raise NotImplementedError(f'unknown implementation {implementation}')
