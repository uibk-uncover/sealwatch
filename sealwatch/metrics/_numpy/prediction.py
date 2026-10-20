"""

Author: Benedikt Lorch, Martin Benes
Affiliation: Universitaet Innsbruck
"""

import numpy as np
from typing import Any

from .common import Meter
from .. import register_meter


class PredictionMeter(Meter):
    """Abstract class to calculate the rolling average value.

    Can be used for additive metrics, such as MSE.
    """
    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, val, n: int = 1):
        """Adds value, averaged over n samples, e.g., the batch size."""
        self._sum += val * n
        self._count += n

    @property
    def avg(self):
        if self._count == 0:
            return np.nan
        return self._sum / self._count


@register_meter(name='loss', backend='numpy', objective='minimize', name_abbr='l')
class LossMeter(PredictionMeter):
    """Computes and stores the average loss value"""
    ...


@register_meter(name='mse', backend='numpy', objective='minimize')
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

    def _error(self, residual):
        """MSE, implemented upon label residuals."""
        return np.nanmean(residual**2)

    def update(self, *ys, mask=None):
        """Updates with the error over the batch.

        :param mask: if the meter has a mask value,
            only elements where mask equals it are considered
        """
        # select the true and predicted vectors
        y_true, y_pred = np.asarray(ys[self.y_true_idx]), np.asarray(ys[self.y_pred_idx])
        # apply mask
        if self.mask is not None:
            if mask is None:
                raise ValueError('meter has a mask value, update requires mask')
            y_true, y_pred = y_true[mask == self.mask], y_pred[mask == self.mask]
        # calculate the error, weighted by the number of valid elements
        residual = y_true - y_pred
        n = int(np.sum(~np.isnan(residual)))
        if n == 0:
            return
        # update the meter
        super().update(self._error(residual), n=n)


@register_meter(name='psnr', backend='numpy', objective='maximize')
class PSNRMeter(MSEMeter):
    """Computes and stores peak signal-to-noise ratio of the average MSE."""
    def __init__(self, *args, max_val: float = 255., **kw):
        super().__init__(*args, **kw)
        self.max_val = max_val

    @property
    def avg(self):
        mse = super().avg
        if mse == 0:
            return np.inf
        return 20 * np.log10(self.max_val / np.sqrt(mse))


@register_meter(name='mae', backend='numpy', objective='minimize')
class MAEMeter(MSEMeter):
    """Computes and stores mean absolute error."""
    def _error(self, residual):
        return np.nanmean(np.abs(residual))


@register_meter(name='ws_mae', backend='numpy', objective='minimize', name_abbr='ws')
class WSMeter(MAEMeter):
    """MAE for relative payload in WS, used to train the U-Net denoiser."""
    def update(self, x, x_hat, alphas):
        """

        :param x: Input cover/stego images.
        :param x_hat: Output, predicted covers.
        :param alphas: Vector of the embedding rates.
        """
        # process input
        x = x[:, :, 1:-1, 1:-1]
        x_hat = x_hat[:, :, 1:-1, 1:-1]
        x_bar = np.round(x).astype('int') ^ 1  # input with flipped LSB

        # compute WS estimate
        weights = np.ones_like(x) / np.prod(x.shape[1:])  # unweighted WS
        betas_hat = np.sum(
            weights * (x - x_bar) * (x - x_hat),  # WS
            axis=(1, 2, 3),
        )
        betas_hat = np.clip(betas_hat, 0, None)  # clip negatives to 0

        # compute reference
        betas = alphas / 2.  # assumed no coding, e=2

        # update with mean absolute error
        super().update(betas, betas_hat)
