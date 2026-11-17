"""Deprecated. The loaders of feature files moved to :mod:`sealwatch.tools`.

Each function warns and calls its counterpart in :mod:`sealwatch.tools`, with the same arguments and results.

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import functools
import warnings

from .. import tools


def _moved(function):
    """Wrap a function of :mod:`sealwatch.tools` under its old name, with a deprecation warning."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"sealwatch.ensemble_classifier.helpers.{function.__name__} is deprecated, "
            f"use sealwatch.tools.{function.__name__}",
            DeprecationWarning,
            stacklevel=2,
        )
        return function(*args, **kwargs)
    wrapper.__doc__ = f"Deprecated. Use :func:`sealwatch.tools.{function.__name__}`."
    return wrapper


load_hdf5 = _moved(tools.load_hdf5)
load_features = _moved(tools.load_features)
remove_file_extension = _moved(tools.remove_file_extension)
load_and_split_features = _moved(tools.load_and_split_features)
load_features_subset = _moved(tools.load_features_subset)
