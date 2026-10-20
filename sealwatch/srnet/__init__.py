"""Deprecated, moved to sealwatch.nn.srnet."""

import sys
import warnings

from ..nn import srnet as _module

warnings.warn(
    'sealwatch.srnet is deprecated, use sealwatch.nn.srnet',
    DeprecationWarning,
    stacklevel=2,
)

# old paths refer to the same modules, e.g., for isinstance and unpickling
sys.modules[__name__] = _module
sys.modules[__name__ + '._srnet'] = _module._srnet
