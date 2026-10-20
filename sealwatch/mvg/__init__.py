"""Multivariate Gaussian (MVG) cover model.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from . import _deflection
from . import _variance

from ._deflection import assumed_rates, deflection, deflection_taylor, deflection_exact
from ._variance import estimate_variance

__all__ = [
    'assumed_rates',
    'deflection',
    'deflection_taylor',
    'deflection_exact',
    'estimate_variance',
]
