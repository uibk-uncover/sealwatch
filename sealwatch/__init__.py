""""""

# attacks
from . import chi2
from . import F5
from . import mvg
from . import rjca
from . import spa
from . import ws

# features
from . import dctr
from . import gfr
from . import hcfcom
from . import jrm
from . import pharm
from . import srm
from . import spam
# from ._sealwatch import spam # Rust-accellerated
from .jrm import ccjrm
from .srm import crm
from .srm import srmq1

# classifier
from . import ensemble_classifier
# neural networks in sealwatch.nn, imported on first use, see __getattr__

#
from . import tools
from . import metrics

# backend to choose
from .tools import get_backend, set_backend
BACKEND_PYTHON = tools.BACKEND_PYTHON
BACKEND_RUST = tools.BACKEND_RUST


#
PHARM_ORIGINAL = pharm.PHARM_ORIGINAL
PHARM_FIX = pharm.PHARM_FIX
PHARM_REVISITED = pharm.PHARM_REVISITED  # legacy name of PHARM_FIX
#
FEATURES_JRM = tools.FEATURES_JRM
FEATURES_CCJRM = tools.FEATURES_CCJRM
FEATURES_SRM = tools.FEATURES_SRM
FEATURES_SRMQ1 = tools.FEATURES_SRMQ1
FEATURES_CRM = tools.FEATURES_CRM
FEATURES_GFR = tools.FEATURES_GFR
FEATURES_PHARM = tools.FEATURES_PHARM
FEATURES_DCTR = tools.FEATURES_DCTR
FEATURES_SPAM = tools.FEATURES_SPAM
#
CRM_ORIGINAL = srm.CRM_ORIGINAL
CRM_FIX_MIN24 = srm.CRM_FIX_MIN24
#
GFR_ORIGINAL = gfr.GFR_ORIGINAL
GFR_FIX = gfr.GFR_FIX
#
WAUC_ALASKA = metrics.WAUC_ALASKA
WAUC_FIX_DIRECTION = metrics.WAUC_FIX_DIRECTION

# package version
import importlib.metadata
try:
    __version__ = importlib.metadata.version("sealwatch")
except importlib.metadata.PackageNotFoundError:
    __version__ = None

__all__ = [
    'crm',
    'ccjrm',
    'chi2',
    'F5',
    'mvg',
    'rjca',
    'spa',
    'ws',
    'tools',
    'dctr',
    'gfr',
    'hcfcom',
    'jrm',
    'pharm',
    'srm',
    'srmq1',
    'spam',
    'ensemble_classifier',
    'metrics',
    '__version__',
]


# moved to sealwatch.nn, deprecated
_NN_MODULES = ('efficientnet', 'srnet', 'ucnet', 'xunet', 'zhunet')
_NN_CLASSES = ('B0', 'B4', 'SRNet', 'UcNet', 'XuNet', 'ZhuNet')
# implementations of PENet, e.g., sw.PENET_FIX
_NN_CONSTANTS = ('PENET_ORIGINAL', 'PENET_FIX')


def __getattr__(name: str):
    """Imports the neural networks on first use, as they require torch."""
    import importlib
    if name == 'nn':
        return importlib.import_module('.nn', __name__)
    if name in _NN_MODULES:
        # the module warns about deprecation
        return importlib.import_module(f'.{name}', __name__)
    if name in _NN_CLASSES:
        import warnings
        warnings.warn(
            f'sealwatch.{name} is deprecated, use sealwatch.nn.{name}',
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module('.nn', __name__), name)
    if name in _NN_CONSTANTS:
        return getattr(importlib.import_module('.nn.penet', __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
