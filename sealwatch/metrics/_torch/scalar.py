"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import torch

from .common import Meter
from .. import register_meter


@register_meter(name='scalar', backend='torch')
class ScalarMeter(Meter):
    """Aggregates per-batch scalar"""
    def __init__(self, *args, key: str, **kw):
        super().__init__(*args, **kw)
        self.key = key
        self.name = key

    def reset(self):
        self._state('scalar', torch.empty(0, device=self.device))
        self._state('batch_size', torch.empty(0, device=self.device))

    def update(self, **ys):
        if self.key not in ys:
            return
        y = ys[self.key]
        if y is None:
            return
        y = y.detach().to(self.device).flatten()

        # batch-level mean as a scalar tensor
        scalar = y.mean().unsqueeze(0)     # shape (1,)
        batch_size = torch.tensor([y.numel()], device=self.device, dtype=torch.float32)

        # append
        self.scalar = torch.cat([self.scalar, scalar])
        self.batch_size = torch.cat([self.batch_size, batch_size])

    @property
    def avg(self):
        total_count = self.batch_size.sum()
        if total_count == 0:
            return float('nan')
        weighted_sum = (self.scalar * self.batch_size).sum()
        return (weighted_sum / total_count).item()
