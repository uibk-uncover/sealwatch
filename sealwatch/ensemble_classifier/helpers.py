"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import h5py
from itertools import compress
import numpy as np
import os
import pandas as pd

from .. import tools
# from sealwatch.utils.logger import setup_custom_logger


log = tools.setup_custom_logger(os.path.basename(__file__))


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


def load_features(cover_features_filename, stego_features_filename, max_num_samples=None):
    """
    Load cover and stego features.

    On the way, drop images where the feature extraction failed. Also drop images where we have no matching cover-stego pairs.

    :param cover_features_filename: path to HDF5 file containing the cover features
    :param stego_features_filename: path to HDF5 file containing the stego features
    :param max_num_samples: take only the first n samples from each dataset. Useful for quick prototyping.
    :return: (cover_features, stego_features, cover_filenames, stego_filenames)
        cover_features and stego_features are ndarrays of shape [num_samples, num_features]
        cover_filenames and stego_filenames are lists with strings
    """
    log.info("Loading cover features")
    cover_features, cover_filenames = load_hdf5(cover_features_filename, max_num_samples=max_num_samples)

    log.info("Loading stego features")
    stego_features, stego_filenames = load_hdf5(stego_features_filename, max_num_samples=max_num_samples)

    num_covers = len(cover_filenames)
    num_stegos = len(stego_filenames)
    assert cover_features.shape[0] == num_covers
    assert stego_features.shape[0] == num_stegos

    # Which filenames are present both in the cover and in the stego set?
    cover_filenames = list([f.decode("utf-8") for f in cover_filenames])
    stego_filenames = list([f.decode("utf-8") for f in stego_filenames])
    # This set is unordered.
    cover_stego_pairs_filenames = set(cover_filenames).intersection(stego_filenames)

    # Drop filenames where the cover feature extraction failed
    drop_cover_mask = np.all(cover_features == 0, axis=1)
    for i in np.where(drop_cover_mask)[0]:
        # In contrast to remove(), discard() does not raise an error if the item is not present in the set.
        cover_stego_pairs_filenames.discard(cover_filenames[i])

    # Drop filenames where the stego feature extraction failed
    drop_stego_mask = np.all(stego_features == 0, axis=1)
    for i in np.where(drop_stego_mask)[0]:
        cover_stego_pairs_filenames.discard(stego_filenames[i])

    # Flag these images where we have no cover-stego pairs
    for i in range(max(num_covers, num_stegos)):
        if i < num_covers:
            drop_cover_mask[i] |= cover_filenames[i] not in cover_stego_pairs_filenames
        if i < num_stegos:
            drop_stego_mask[i] |= stego_filenames[i] not in cover_stego_pairs_filenames

    num_covers_dropped = np.sum(drop_cover_mask)
    num_stegos_dropped = np.sum(drop_stego_mask)
    log.info(f"Dropping {num_covers_dropped} cover and {num_stegos_dropped} stego images that do not exist in the other set.")

    # Synchronize order and drop images
    cover_permutation = np.argsort(cover_filenames)
    cover_permutation = cover_permutation[~drop_cover_mask]
    cover_filenames = [cover_filenames[i] for i in cover_permutation]
    cover_features = cover_features[cover_permutation]

    stego_permutation = np.argsort(stego_filenames)
    stego_permutation = stego_permutation[~drop_stego_mask]
    stego_filenames = [stego_filenames[i] for i in stego_permutation]
    stego_features = stego_features[stego_permutation]
    assert np.all(cover_filenames == stego_filenames)
    assert cover_features.shape == stego_features.shape

    # Check for matching cover and stego samples
    cover_stego_features_match = np.all(np.isclose(cover_features, stego_features), axis=1)
    num_matching_samples = np.sum(cover_stego_features_match)
    log.info(f"Sanity check: {num_matching_samples}/{len(cover_features)} and stego image features are identical.")
    if num_matching_samples > 0:
        matching_filenames = [cover_filenames[i] for i in np.where(cover_stego_features_match)[0]]
        log.info(f"Matching images: " + ", ".join(matching_filenames))

    return cover_features, stego_features, cover_filenames, stego_filenames


def remove_file_extension(f):
    return os.path.splitext(f)[0]


def load_and_split_features(cover_features_filename, stego_features_filename, train_csv, test_csv, max_num_samples=None):
    """
    Load cover and stego features and split them into training and test sets.

    On the way, drop images where the feature extraction failed. Also drop images where we have no matching cover-stego pairs.

    :param cover_features_filename: path to HDF5 file containing the cover features
    :param stego_features_filename: path to HDF5 file containing the stego features
    :param train_csv: csv file containing the filenames to use for training
    :param test_csv: csv file containing the filenames to use for testing
    :param max_num_samples: take only the first n samples from each dataset. Useful for quick prototyping.
    :return: 6-tuple
        0: cover_features_train,
        1: stego_features_train,
        2: cover_features_test,
        3: stego_features_test
        4: train_filenames (same lengths as covers; the covers and stegos have the same filenames)
        5: test_filenames (same length as covers; the covers and stegos have the same filename)
    """
    cover_features, stego_features, cover_filenames, stego_filenames = load_features(
        cover_features_filename=cover_features_filename,
        stego_features_filename=stego_features_filename,
        max_num_samples=max_num_samples,
    )

    # Remove file extensions
    cover_filenames = list(map(remove_file_extension, cover_filenames))
    stego_filenames = list(map(remove_file_extension, stego_filenames))

    # Split into train and test sets
    log.info("Splitting into training and test sets")

    # Load training split
    train_df = pd.read_csv(train_csv)
    assert "filename" in train_df.columns, f"Expected column \"filename\" in \"{train_csv}\""
    # Remove file extension
    train_df["filename_no_ext"] = train_df["filename"].map(remove_file_extension)
    train_filenames = set(train_df["filename_no_ext"])

    # Load test split
    test_df = pd.read_csv(test_csv)
    assert "filename" in test_df.columns, f"Expected column \"filename\" in \"{test_csv}\""
    # Remove file extension
    test_df["filename_no_ext"] = test_df["filename"].map(remove_file_extension)
    test_filenames = set(test_df["filename_no_ext"])

    # Set up empty mask
    train_mask = np.zeros(len(cover_features), dtype=bool)
    test_mask = np.zeros(len(cover_features), dtype=bool)

    # At this point, cover_filenames == stego_filenames
    for i, filename in enumerate(cover_filenames):
        train_mask[i] = filename in train_filenames
        test_mask[i] = filename in test_filenames
        assert not (train_mask[i] and test_mask[i]), "Training and test sets should be mutually exclusive"

    # Note that some images may not be present in training and test set, e.g., when there is a third set.

    # Split features into training and test subsets
    cover_features_train = cover_features[train_mask]
    stego_features_train = stego_features[train_mask]

    cover_features_test = cover_features[test_mask]
    stego_features_test = stego_features[test_mask]

    # Filenames
    train_filenames_sorted = list(compress(cover_filenames, train_mask))
    test_filenames_sorted = list(compress(cover_filenames, test_mask))

    if len(train_filenames_sorted) == 0:
        log.warning("Retained 0 training images. Please check whether the filenames in your csv file match the filenames in your features file.")

    if len(test_filenames_sorted) == 0:
        log.warning("Retained 0 test images. Please check whether the filenames in your csv file match the filenames in your features file.")

    return cover_features_train, stego_features_train, cover_features_test, stego_features_test, train_filenames_sorted, test_filenames_sorted


def load_features_subset(cover_features_filename, stego_features_filename, test_csv):
    cover_features, stego_features, cover_filenames, stego_filenames = load_features(
        cover_features_filename=cover_features_filename,
        stego_features_filename=stego_features_filename,
    )

    assert cover_filenames == stego_filenames
    assert cover_features.shape == stego_features.shape

    test_df = pd.read_csv(test_csv)
    test_filenames = set(test_df["filename"])

    test_mask = np.zeros(len(cover_features), dtype=bool)
    for i, filename in enumerate(cover_filenames):
        test_mask[i] = filename in test_filenames

    cover_features_test = cover_features[test_mask]
    stego_features_test = stego_features[test_mask]
    test_filenames_synchronized = list(compress(cover_filenames, test_mask))

    return cover_features_test, stego_features_test, test_filenames_synchronized
