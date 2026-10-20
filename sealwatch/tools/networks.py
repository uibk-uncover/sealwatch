"""Deprecated, moved to sealwatch.nn."""

import sys
import warnings

from ..nn import _common as _module

warnings.warn(
    'sealwatch.tools.networks is deprecated, use sealwatch.nn',
    DeprecationWarning,
    stacklevel=2,
)

# old path refers to the same module
sys.modules[__name__] = _module
