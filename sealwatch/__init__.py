
import pkg_resources

# attacks
from . import chi2
from . import ws

# features

#
from . import tools

# package version
try:
    __version__ = pkg_resources.get_distribution("sealwatch").version
except pkg_resources.DistributionNotFound:
    __version__ = None

__all__ = [
    'chi2',
    'ws',
    'tools',
    '__version__',
]
