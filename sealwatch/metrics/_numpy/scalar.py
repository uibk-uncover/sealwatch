"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np

from .common import Meter
from .. import register_meter


@register_meter(name='scalar', backend='numpy')
class ScalarMeter(Meter):
    """Abstract class for scalar meter.

    Can be used for metrics that aggregate over batch to report a scalar.
    """

    def __init__(self, *args, key: str, **kw):
        super().__init__(*args, **kw)
        self.key = key

    def reset(self):
        self.scalar = np.array([])
        self.batch_size = np.array([])

    def update(self, **ys):
        y = ys[self.key]
        scalar = np.mean(y, keepdims=True)
        batch_size = np.array([np.size(y)])
        self.scalar = np.concatenate((self.scalar, scalar))
        self.batch_size = np.concatenate((self.batch_size, batch_size))

    @property
    def avg(self):
        if np.sum(self.batch_size) == 0:
            return np.nan
        return np.sum(self.scalar * self.batch_size) / np.sum(self.batch_size)
