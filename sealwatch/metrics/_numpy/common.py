"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from abc import ABC, abstractmethod


class Meter(ABC):
    """Interface for meter classes."""
    name: str
    backend: str
    objective: str
    name_abbr: str

    def __init__(self, fmt: str = ':4.3f'):
        """"""
        self.fmt = fmt
        self.reset()

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
