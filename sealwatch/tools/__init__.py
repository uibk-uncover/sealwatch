"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

from . import _backend
from . import calibration
from . import dct
from . import features
from . import hpf
from . import jpeg
from . import matlab
from . import signal
#
from ._defs import setup_custom_logger, BufferedWriter, EPS
from ._backend import Backend, set_backend, get_backend, rust_module
from .features import flatten, group, Features
from .dct import jpeglib_to_jpegio, jpegio_to_jpeglib
from ._zarr import (
    create_feature_store,
    feature_task_rows,
    num_feature_tasks,
    write_feature_task,
    missing_feature_tasks,
    load_zarr,
)
from ._hdf5 import load_hdf5
from ._loaders import (
    load_features,
    load_and_split_features,
    load_features_subset,
    remove_file_extension,
)

#
BACKEND_PYTHON = Backend.BACKEND_PYTHON
BACKEND_RUST = Backend.BACKEND_RUST
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
    'backend',
    'Backend',
    'calibration',
    'dct',
    'features',
    'hpf',
    'jpeg',
    'matlab',
    'signal',
    'setup_custom_logger',
    'BufferedWriter',
    'EPS',
    'flatten',
    'group',
    'jpeglib_to_jpegio',
    'jpegio_to_jpeglib',
    'create_feature_store',
    'feature_task_rows',
    'num_feature_tasks',
    'write_feature_task',
    'missing_feature_tasks',
    'load_zarr',
    'load_hdf5',
    'load_features',
    'load_and_split_features',
    'load_features_subset',
    'remove_file_extension',
]


def __getattr__(name: str):
    """Imports the deprecated networks on first use, as they require torch."""
    if name == 'networks':
        import importlib
        return importlib.import_module('.networks', __name__)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
