"""Deprecated, moved to sealwatch.nn.ucnet."""

import sys
import warnings

from ..nn import ucnet as _module

warnings.warn(
    'sealwatch.ucnet is deprecated, use sealwatch.nn.ucnet',
    DeprecationWarning,
    stacklevel=2,
)

# old paths refer to the same modules, e.g., for isinstance and unpickling
sys.modules[__name__] = _module
sys.modules[__name__ + '._ucnet'] = _module._ucnet
