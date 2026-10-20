"""Deprecated, moved to sealwatch.nn.efficientnet."""

import sys
import warnings

from ..nn import efficientnet as _module

warnings.warn(
    'sealwatch.efficientnet is deprecated, use sealwatch.nn.efficientnet',
    DeprecationWarning,
    stacklevel=2,
)

# old paths refer to the same modules, e.g., for isinstance and unpickling
sys.modules[__name__] = _module
sys.modules[__name__ + '._efficientnet'] = _module._efficientnet
