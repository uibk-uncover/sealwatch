"""

Author: Benedikt Lorch, Martin Benes
Affiliation: Universitaet Innsbruck
"""

import torch
from typing import Any, Tuple, Union

from .common import Meter
from .. import register_meter


class PredictionMeter(Meter):
    """Abstract class to calculate the average value.

    Can be used for additive metrics, such as MSE.
    By default, it is the mean weighted by the number of samples.
    With gamma, it is the exponential moving average instead.
    """
    def __init__(self, *args, gamma: float = None, **kw):
        self.gamma = gamma
        super().__init__(*args, **kw)

    def reset(self):
        # kept on device, to avoid synchronization at each update
        self._state('_sum', torch.zeros((), device=self.device, dtype=torch.float32))
        self._state('_sum2', torch.zeros((), device=self.device, dtype=torch.float32))
        self._state('_count', torch.zeros((), device=self.device, dtype=torch.float32))

    def update(self, val: Union[torch.Tensor, float], n: int = 1):
        """Adds value, averaged over n samples, e.g., the batch size."""
        val = torch.as_tensor(val).detach().to(device=self.device, dtype=torch.float32)
        if self.gamma is None:
            self._sum = self._sum + val * n
            self._sum2 = self._sum2 + val ** 2 * n
            self._count = self._count + n
        elif self._count == 0:
            self._sum, self._sum2, self._count = val, val ** 2, self._count + 1
        else:
            # exponential moving average, _sum holds the mean
            self._sum = self.gamma * self._sum + (1 - self.gamma) * val
            self._sum2 = self.gamma * self._sum2 + (1 - self.gamma) * val ** 2
            self._count = self._count + 1

    def _mean(self):
        if self.gamma is None:
            return self._sum / self._count, self._sum2 / self._count
        return self._sum, self._sum2

    @property
    def avg(self):
        if self._count == 0:
            return float('nan')
        return self._mean()[0].item()

    @property
    def var(self):
        if self._count == 0:
            return float('nan')
        mean, mean2 = self._mean()
        return (mean2 - mean**2).item()


@register_meter(name='loss', backend='torch', objective='minimize', name_abbr='l')
class LossMeter(PredictionMeter):
    """Computes and stores the average loss value"""


@register_meter(name='mse', backend='torch', objective='minimize')
class MSEMeter(PredictionMeter):
    """Computes and stores mean squared error."""
    def __init__(
        self,
        *args,
        y_true_idx: int = 0,
        y_pred_idx: int = 1,
        mask: Any = None,
        **kw,
    ):
        super().__init__(*args, **kw)
        self.mask = mask
        self.y_true_idx = y_true_idx
        self.y_pred_idx = y_pred_idx

    def compute(self, pred: torch.Tensor, target: torch.Tensor):
        return torch.nanmean((pred - target) ** 2)

    def update(self, *ys: Tuple[torch.Tensor], mask: torch.Tensor = None):
        """Updates with the error over the batch.

        :param mask: if the meter has a mask value,
            only elements where mask equals it are considered
        """
        # select the true and predicted vectors
        y_true, y_pred = ys[self.y_true_idx].detach(), ys[self.y_pred_idx].detach()
        # apply mask
        if self.mask is not None:
            if mask is None:
                raise ValueError('meter has a mask value, update requires mask')
            y_true, y_pred = y_true[mask == self.mask], y_pred[mask == self.mask]
        # calculate the error, weighted by the number of valid elements
        n = (~torch.isnan(y_pred - y_true)).sum()
        error = self.compute(y_pred, y_true)
        # update the meter, without synchronization
        super().update(torch.nan_to_num(error), n=n)


@register_meter(name='psnr', backend='torch', objective='maximize')
class PSNRMeter(MSEMeter):
    """Computes and stores peak signal-to-noise ratio of the average MSE."""
    def __init__(self, *args, max_val: float = 255., **kw):
        super().__init__(*args, **kw)
        self.max_val = max_val

    @property
    def avg(self):
        mse = super().avg
        if mse == 0:
            return float('inf')
        return (20 * torch.log10(self.max_val / torch.sqrt(torch.tensor(mse)))).item()


@register_meter(name='mae', backend='torch', objective='minimize')
class MAEMeter(MSEMeter):
    def compute(self, pred: torch.Tensor, target: torch.Tensor):
        return torch.nanmean(torch.abs(pred - target))
