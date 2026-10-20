"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from abc import ABC, abstractmethod
import torch


class Meter(ABC, torch.nn.Module):
    """Interface for meter classes.

    The state is kept in non-persistent buffers,
    so that .to() moves it along with the meter.
    """
    name: str
    backend: str
    objective: str
    name_abbr: str

    def __init__(self, fmt: str = ':4.3f', device: torch.device = None):
        """"""
        super().__init__()
        # empty buffer, tracks the device of the meter
        self.register_buffer('_anchor', torch.empty(0, device=device), persistent=False)
        self.fmt = fmt
        self.reset()

    @property
    def device(self) -> torch.device:
        return self._anchor.device

    def _state(self, name: str, value: torch.Tensor = None):
        """Registers (or resets) a state tensor."""
        self.register_buffer(name, value, persistent=False)

    @abstractmethod
    def reset(self):
        """"""
        pass

    @abstractmethod
    def update(self, *args, **kw):
        """"""
        pass

    @property
    @abstractmethod
    def avg(self):
        """"""
        pass

    def __str__(self):
        """"""
        fmtstr = '{name}={avg' + self.fmt + '}'
        return fmtstr.format(name=self.name, avg=self.avg)

    def to_dict(self):
        """"""
        return {self.name: self.avg}
