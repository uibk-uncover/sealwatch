"""

Authors: Benedikt Lorch
Affiliation: University of Innsbruck
"""

from typing import Any

from . import pharm_original
from . import pharm_revisited
from ._pharm import extract, extract_from_file, Implementation, qf_to_quantization_step, select_quantization_step


PHARM_ORIGINAL = Implementation.PHARM_ORIGINAL
PHARM_FIX = Implementation.PHARM_FIX
PHARM_REVISITED = Implementation.PHARM_REVISITED  # legacy name of PHARM_FIX


def get_extractor_factory(
    implementation: Implementation = Implementation.PHARM_FIX,
) -> Any:  # Extractor superclass
    """Class of the former implementation, kept for now. Use extract()."""
    if implementation is Implementation.PHARM_FIX:
        construct_extractor = pharm_revisited.RevisitedExtractor
    elif implementation is Implementation.PHARM_ORIGINAL:
        construct_extractor = pharm_original.OriginalExtractor
    else:
        raise NotImplementedError(f'unknown implementation {implementation}')
    return construct_extractor
