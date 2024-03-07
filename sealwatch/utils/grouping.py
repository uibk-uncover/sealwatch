import numpy as np
from collections import OrderedDict
from sealwatch.features.jrm import extract_cc_jrm_features_from_file, extract_jrm_features_from_file
from sealwatch.features.gfr import extract_gfr_features_from_file
from sealwatch.features.pharm import extract_pharm_original_features_from_file, extract_pharm_revisited_features_from_file
from sealwatch.features.spam import extract_spam686_features_from_file
from sealwatch.utils.constants import JRM, CC_JRM, GFR, PHARM_ORIGINAL, PHARM_REVISITED, SPAM
import tempfile
import jpeglib


def generate_groups(feature_type):
    img = np.random.randint(low=0, high=256, size=(64, 64, 1), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".jpeg") as f:
        jpeglib.from_spatial(img).write_spatial(f.name, qt=75)

        if JRM == feature_type:
            dummy_features = extract_jrm_features_from_file(f.name)

        elif CC_JRM == feature_type:
            dummy_features = extract_cc_jrm_features_from_file(f.name)

        elif GFR == feature_type:
            dummy_features = extract_gfr_features_from_file(f.name, num_rotations=32, qf=75)

        elif PHARM_ORIGINAL == feature_type:
            dummy_features = extract_pharm_original_features_from_file(f.name)

        elif PHARM_REVISITED == feature_type:
            dummy_features = extract_pharm_revisited_features_from_file(f.name)

        elif SPAM == feature_type:
            dummy_features = extract_spam686_features_from_file(f.name)

        else:
            raise ValueError("Unknown feature type")

    return dummy_features


def group_single(features, feature_type):
    """
    Split a feature vector into groups
    :param features: ndarray of shape [num_features]
    :param feature_type: type of features
    :return: ordered dict, where the keys are the submodel names
    """

    assert len(features.shape) == 1

    total_num_features = len(features)

    # Extract features from a dummy image. This gives us the submodel names and associated dimensionality for slicing the given input features.
    dummy_groups = generate_groups(feature_type)

    # Groups to return
    groups = OrderedDict()

    # Iterate over submodels
    next_col = 0
    for submodel_name, dummy_features in dummy_groups.items():
        num_features = len(dummy_features.flatten())

        # Take next slice from the input features and assign the submodel name
        groups[submodel_name] = features[..., next_col:next_col + num_features].reshape(dummy_features.shape)

        next_col += num_features

    assert next_col == total_num_features, "Mismatch in the total number of features"

    return groups


def group_batch(features, feature_type):
    """
    Split a flat feature vector into groups
    :param features: ndarray of shape [num_samples, num_features]
    :param feature_type: type of features
    :return: ordered dict, where the keys are the submodel names
    """

    num_samples, total_num_features = features.shape

    # Extract features from a dummy image. This gives us the submodel names and associated dimensionality for slicing the given input features.
    dummy_groups = generate_groups(feature_type)

    # Groups to return
    groups = OrderedDict()

    # Iterate over submodels
    next_col = 0
    for submodel_name, dummy_features in dummy_groups.items():
        num_features = len(dummy_features.flatten())

        # Take next slice from the input features and assign the submodel name
        # Reshape to [num_samples, submodel_shape]
        group_shape = (num_samples,) + dummy_features.shape
        groups[submodel_name] = features[..., next_col:next_col + num_features].reshape(group_shape)

        next_col += num_features

    assert next_col == total_num_features, "Mismatch in the total number of features"

    return groups


def flatten_single(grouped_features):
    """
    Flatten the submodels of a single sample to an ndarray.
    :param grouped_features: ordered dict, where the keys represent the submodel names and the values are the submodels
    :return: 1D array of length [num_features]
    """

    submodel_features = grouped_features.values()
    return np.concatenate([f.flatten() for f in submodel_features])


def flatten_batch(batch_grouped_features):
    """
    Flatten a batch of grouped features to an ndarray.
    :param batch_grouped_features: ordered dict, where the keys represent the submodel names and the values are the submodels, with one row per sample.
    :return: 2D array of shape [num_samples, num_features]
    """

    # Batch size
    num_samples = None
    dtype = None

    # Determine the total number of features
    num_features = 0
    for submodel_features_batch in batch_grouped_features.values():
        submodel_num_samples = submodel_features_batch.shape[0]

        # Verify that the number of samples matches the other submodels
        if num_samples is None:
            num_samples = submodel_num_samples
            dtype = submodel_features_batch.dtype
        else:
            assert num_samples == submodel_num_samples, "Mismatch in the number of samples"
            assert dtype == submodel_features_batch.dtype, "Mismatch in dtype"

        num_features += np.prod(submodel_features_batch.shape[1:])

    # Allocate output array
    features = np.zeros((num_samples, num_features), dtype=dtype)

    # Set up column pointer
    next_col = 0

    for submodel_features_batch in batch_grouped_features.values():
        # Calculate number of features in current group
        submodel_num_features = np.prod(submodel_features_batch.shape[1:])

        # Copy into output array
        features[:, next_col:next_col + submodel_num_features] = submodel_features_batch.reshape(num_samples, submodel_num_features)

        # Advance column pointer
        next_col += submodel_num_features

    return features
