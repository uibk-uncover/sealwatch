"""Feature files in HDF5.

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import os

import h5py
import numpy as np

from ._defs import setup_custom_logger


log = setup_custom_logger(os.path.basename(__file__))


def load_hdf5(features_filename, max_num_samples=None):
    """
    Retrieve features and filenames from a HDF5 file.

    When the origin attribute is "matlab", the feature array is transposed.

    :param features_filename: path to HDF5 file
    :param max_num_samples: Only load the first n features and filenames
    :return: (features, filenames) as 2-tuple
    """
    with h5py.File(features_filename, "r") as f:
        features_ds = f["features"]
        filenames_ds = f["filenames"]

        origin = f.attrs.get("origin")
        if origin == "matlab":
            if max_num_samples:
                log.warning(f"Taking only the first {max_num_samples} samples")
                features = features_ds[:, :max_num_samples]
                filenames = filenames_ds[:max_num_samples]
            else:
                features = features_ds[()]
                filenames = filenames_ds[()]

            # Feature arrays exported from Matlab need to be transposed
            features = np.transpose(features)

        else:
            # No Matlab => no need to transpose
            if max_num_samples:
                log.warning(f"Taking only the first {max_num_samples} samples")
                features = features_ds[:max_num_samples, :]
                filenames = filenames_ds[:max_num_samples]
            else:
                features = features_ds[()]
                filenames = filenames_ds[()]

    return features, filenames
