
import pkg_resources

# attacks
from . import chi2
from . import rjca
from . import spa
from . import ws

# features

#
from . import utils

# package version
try:
    __version__ = pkg_resources.get_distribution("sealwatch").version
except pkg_resources.DistributionNotFound:
    __version__ = None

__all__ = [
    'chi2',
    'rjca',
    'ws',
    'utils',
    '__version__',
]
