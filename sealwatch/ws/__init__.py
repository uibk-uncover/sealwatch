"""

Authors: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

from . import _attack

from ._attack import attack


def __getattr__(name: str):
    """Imports the U-Net pixel predictor on first use, as it requires torch."""
    import importlib
    if name == '_unet':
        return importlib.import_module('._unet', __name__)
    if name in ('unet_estimator', 'UNet'):
        return getattr(importlib.import_module('._unet', __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
