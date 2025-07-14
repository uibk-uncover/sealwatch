"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

from . import calibration
from . import dct
from . import features
from . import jpeg
from . import matlab
from . import networks
from . import signal
#
from ._defs import setup_custom_logger, BufferedWriter, EPS
from .features import flatten, group, Features
from .dct import jpeglib_to_jpegio, jpegio_to_jpeglib

#
FEATURES_JRM = Features.FEATURES_JRM
FEATURES_CCJRM = Features.FEATURES_CCJRM
FEATURES_SRM = Features.FEATURES_SRM
FEATURES_SRMQ1 = Features.FEATURES_SRMQ1
FEATURES_CRM = Features.FEATURES_CRM
FEATURES_GFR = Features.FEATURES_GFR
FEATURES_PHARM = Features.FEATURES_PHARM
FEATURES_DCTR = Features.FEATURES_DCTR
FEATURES_SPAM = Features.FEATURES_SPAM

__all__ = [
    'calibration',
    'dct',
    'features',
    'jpeg',
    'matlab',
    'networks',
    'signal',
    'setup_custom_logger',
    'BufferedWriter',
    'EPS',
    'flatten',
    'group',
    'jpeglib_to_jpegio',
    'jpegio_to_jpeglib',
]