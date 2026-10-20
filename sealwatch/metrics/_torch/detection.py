"""Detection meters for torch, delegating to the numpy implementation.

The labels and scores are accumulated on their device, without synchronization.
When the value is requested, they are moved to CPU once,
and the metric is computed by the corresponding numpy meter.

Author: Benedikt Lorch, Martin Benes
Affiliation: Universitaet Innsbruck
"""

import functools
import numpy as np
import torch
from typing import Any, Callable

from .common import Meter
from .. import register_meter
from .._numpy import detection as _numpy


def _to_numpy(x: Any) -> Any:
    """Converts tensor to float64 numpy array, other values unchanged."""
    if isinstance(x, torch.Tensor):
        # to CPU first, as some devices do not support float64
        return x.detach().cpu().to(torch.float64).numpy()
    return x


def _to_torch(x: Any) -> Any:
    """Converts numpy array to tensor, numpy scalar to float, recursively in tuples."""
    if isinstance(x, tuple):
        return tuple(_to_torch(xi) for xi in x)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, np.generic):
        return x.item()
    return x


def _wrap(fn: Callable) -> Callable:
    """Calls numpy function on tensors."""
    @functools.wraps(fn)
    def wrapped(*args, **kw):
        args = [_to_numpy(a) for a in args]
        kw = {k: _to_numpy(v) for k, v in kw.items()}
        return _to_torch(fn(*args, **kw))
    return staticmethod(wrapped)


class DetectionMeter(Meter):
    """Abstract class for detection meter.

    Can be used for metrics that need all the labels and predictions.
    Such metrics are typically derived from the ROC curve or confusion matrix.
    """
    numpy_cls: type = None

    def __init__(
        self,
        *args,
        y_true_idx: int = 0,
        y_pred_idx: int = 1,
        history: int = None,
        fmt: str = ':4.3f',
        device: torch.device = None,
        **kw,
    ):
        self.y_true_idx = y_true_idx
        self.y_pred_idx = y_pred_idx
        self.history = history
        # parameters of the metric, e.g., threshold or fpr_level
        self._numpy_kw = kw
        self.numpy_cls(**kw)  # validate parameters
        super().__init__(fmt=fmt, device=device)

    def __getattr__(self, name: str):
        """Parameters of the metric, e.g., meter.threshold."""
        numpy_kw = self.__dict__.get('_numpy_kw', {})
        if name in numpy_kw:
            return numpy_kw[name]
        return super().__getattr__(name)

    def reset(self):
        self._y_true, self._y_pred, self._count = [], [], 0

    def update(self, *ys):
        """Accumulates labels and scores, on their device."""
        y_true = ys[self.y_true_idx].detach().flatten()
        y_pred = ys[self.y_pred_idx].detach().flatten()
        if len(y_true) != len(y_pred):
            raise ValueError(f'got {len(y_true)} labels and {len(y_pred)} predictions')
        self._y_true.append(y_true)
        self._y_pred.append(y_pred)
        self._count += len(y_true)
        # drop batches out of the history
        if self.history:
            while self._count - len(self._y_true[0]) >= self.history:
                self._count -= len(self._y_true.pop(0))
                self._y_pred.pop(0)

    def _concat(self, ys):
        if not ys:
            return torch.empty(0)
        y = torch.cat([yi.to(ys[-1].device) for yi in ys])
        return y[-self.history:] if self.history else y

    @property
    def y_true(self) -> torch.Tensor:
        return self._concat(self._y_true)

    @property
    def y_pred(self) -> torch.Tensor:
        return self._concat(self._y_pred)

    def to_numpy(self):
        """Returns the numpy meter with the accumulated labels and scores."""
        meter = self.numpy_cls(**self._numpy_kw)
        meter.y_true = _to_numpy(self.y_true)
        meter.y_pred = _to_numpy(self.y_pred)
        return meter

    @property
    def avg(self) -> float:
        return float(self.to_numpy().avg)


def _delegate(numpy_cls: type) -> type:
    """Creates and registers torch meter, delegating to numpy meter."""
    namespace = {
        'numpy_cls': numpy_cls,
        '__doc__': numpy_cls.__doc__,
        '__module__': __name__,
    }
    # functions of the metric, e.g., roc or optimal_pe
    for attr in ('roc', 'roc_quantile', 'given_pe', 'optimal_pe', 'auc', 'fnr_at_fpr', 'wauc_fix_direction', 'wauc_alaska'):
        if hasattr(numpy_cls, attr):
            namespace[attr] = _wrap(getattr(numpy_cls, attr))
    cls = type(numpy_cls.__name__, (DetectionMeter,), namespace)
    return register_meter(
        name=numpy_cls.name,
        backend='torch',
        objective=numpy_cls.objective,
        name_abbr=numpy_cls.name_abbr if numpy_cls.name_abbr != numpy_cls.name else None,
    )(cls)


AccuracyMeter = _delegate(_numpy.AccuracyMeter)
MisclassificationMeter = _delegate(_numpy.MisclassificationMeter)
PrecisionMeter = _delegate(_numpy.PrecisionMeter)
RecallMeter = _delegate(_numpy.RecallMeter)
PEMeter = _delegate(_numpy.PEMeter)
AUCMeter = _delegate(_numpy.AUCMeter)
wAUCMeter = _delegate(_numpy.wAUCMeter)
FNRatFPRMeter = _delegate(_numpy.FNRatFPRMeter)
