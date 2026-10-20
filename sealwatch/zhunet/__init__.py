"""Deprecated, moved to sealwatch.nn.zhunet."""

import sys
import warnings

from ..nn import zhunet as _module

warnings.warn(
    'sealwatch.zhunet is deprecated, use sealwatch.nn.zhunet',
    DeprecationWarning,
    stacklevel=2,
)

# old paths refer to the same modules, e.g., for isinstance and unpickling
sys.modules[__name__] = _module
sys.modules[__name__ + '._zhunet'] = _module._zhunet
