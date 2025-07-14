""""""

# attacks
from . import chi2
from . import F5
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
from ._sealwatch import spam_rs # Rust-accellerated
from .jrm import ccjrm
from .srm import crm
from .srm import srmq1

# classifier
from . import ensemble_classifier
from . import xunet

#
from . import tools


#
PHARM_ORIGINAL = pharm.PHARM_ORIGINAL
PHARM_REVISITED = pharm.PHARM_REVISITED
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
    'rjca',
    'spa',
    'ws',
    'tools',
    'features',
    'dctr',
    'gfr',
    'hcfcom',
    'jrm',
    'pharm',
    'srm',
    'srmq1',
    'spam',
    'spam_rs',
    'ensemble_classifier',
    'xunet',
    '__version__',
]
