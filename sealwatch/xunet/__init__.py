"""Deprecated, moved to sealwatch.nn.xunet."""

import sys
import warnings

from ..nn import xunet as _module

warnings.warn(
    'sealwatch.xunet is deprecated, use sealwatch.nn.xunet',
    DeprecationWarning,
    stacklevel=2,
)

# old paths refer to the same modules, e.g., for isinstance and unpickling
sys.modules[__name__] = _module
sys.modules[__name__ + '._xunet'] = _module._xunet
