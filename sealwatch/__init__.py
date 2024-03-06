
import pkg_resources

# attacks
from . import chi2

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
    'tools',
    '__version__',
]
