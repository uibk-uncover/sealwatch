
import numpy as np

from . import calibration
from . import constants
from . import convolution
from . import dct
from . import dict
from . import grouping
from . import jpeg
from . import logger
from . import matlab
from . import quantization_table
from . import writer

from .dct import jpeglib_to_jpegio, jpegio_to_jpeglib
from .grouping import flatten_single

EPS = np.finfo(np.float64).eps
"""small numerical constant"""