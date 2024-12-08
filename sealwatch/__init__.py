
import pkg_resources

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

# classifier
from . import ensemble_classifier

#
from . import tools

#
PHARM_ORIGINAL = pharm.Implementation.PHARM_ORIGINAL
PHARM_REVISITED = pharm.Implementation.PHARM_REVISITED
#
FEATURES_JRM = tools.grouping.Features.FEATURES_JRM
FEATURES_CCJRM = tools.grouping.Features.FEATURES_CCJRM
FEATURES_SRM = tools.grouping.Features.FEATURES_SRM
FEATURES_CRM = tools.grouping.Features.FEATURES_CRM
FEATURES_GFR = tools.grouping.Features.FEATURES_GFR
FEATURES_PHARM = tools.grouping.Features.FEATURES_PHARM
FEATURES_SPAM = tools.grouping.Features.FEATURES_SPAM

# package version
try:
    __version__ = pkg_resources.get_distribution("sealwatch").version
except pkg_resources.DistributionNotFound:
    __version__ = None

__all__ = [
    'chi2',
    'rjca',
    'ws',
    'tools',
    'features',
    'dctr',
    'gfr',
    'hcfcom',
    'jrm',
    'pharm',
    'spam',
    '__version__',
]
