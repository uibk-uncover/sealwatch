
import enum


class Implementation(enum.Enum):
    """Metric implementation to choose from."""

    WAUC_ALASKA = enum.auto()
    """Weighted AUC of the ALASKA2 competition, splitting the ROC horizontally."""
    WAUC_FIX_DIRECTION = enum.auto()
    """Weighted AUC, splitting the ROC vertically."""


WAUC_ALASKA = Implementation.WAUC_ALASKA
WAUC_FIX_DIRECTION = Implementation.WAUC_FIX_DIRECTION

METER_REGISTRY = {}

def register_meter(name, backend, objective: str = None, name_abbr: str = None):
    def decorator(cls):
        for attr in ('name', 'backend', 'objective'):
            if attr in cls.__dict__:
                raise AttributeError(
                    f"{cls.__name__} defines '{attr}' explicitly; "
                    "remove it and use @register_meter"
                )
        keys = [(name, backend)]
        if name_abbr:
            keys.append((name_abbr, backend))
        for key in keys:
            if key in METER_REGISTRY:
                raise KeyError(f"Meter already registered: {key}")
        for key in keys:
            METER_REGISTRY[key] = cls
        cls.name = name
        cls.backend = backend
        cls.objective = objective
        if name_abbr:
            cls.name_abbr = name_abbr
        else:
            cls.name_abbr = name
        return cls
    return decorator


from . import _numpy


def __getattr__(name: str):
    """Imports the torch backend on first use, as it requires torch."""
    if name == '_torch':
        import importlib
        return importlib.import_module('._torch', __name__)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def get_meter(name, backend: str = 'torch', **kw):
    if backend == 'torch':
        __getattr__('_torch')  # registers the torch meters
    try:
        cls = METER_REGISTRY[(name, backend)]
    except KeyError:
        raise ValueError(f"No meter '{name}' for backend '{backend}'")
    return cls(**kw)
