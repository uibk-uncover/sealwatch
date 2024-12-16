"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

from collections import OrderedDict
import enum
import numpy as np

from . import dct
from .. import jrm, gfr, pharm, spam, srm


class Features(enum.Enum):
    """Features."""

    FEATURES_JRM = enum.auto()
    """JRM features."""
    FEATURES_CCJRM = enum.auto()
    """CC-JRM features."""
    FEATURES_SRM = enum.auto()
    """SRM features."""
    FEATURES_SRMQ1 = enum.auto()
    """SRM features."""
    FEATURES_CRM = enum.auto()
    """CRM features."""
    FEATURES_GFR = enum.auto()
    """GFR features."""
    FEATURES_PHARM = enum.auto()
    """PHARM features."""
    FEATURES_SPAM = enum.auto()
    """SPAM features."""
    FEATURES_DCTR = enum.auto()
    """DCTR features."""


def append(
    target: OrderedDict,
    source: OrderedDict,
    prefix: str = None,
) -> OrderedDict:
    """Copy features from a source dict to a target dict. Prepend a prefix to the source's keys before copying.
    :param target: (ordered) dict
    :param source: (ordered) dict
    :param prefix: append this prefix to the target key name
    :return: target
    :rtype:

    :Example:

    >>> # TODO
    """
    assert isinstance(target, OrderedDict), "Expected target to be a dict"
    assert isinstance(source, OrderedDict), "Expected source to be a adict"

    for source_key, value in source.items():
        if prefix is None:
            target_key = source_key
        else:
            target_key = prefix + "_" + source_key

        target[target_key] = value

    return target


def generate_groups(
    feature_type: Features,
) -> OrderedDict:
    """

    :param feature_type:
    :type feature_type:
    :return:
    :rtype:

    :Example:

    >>> # TODO
    """
    # generate foo DCT
    rng = np.random.default_rng(12345)
    x = rng.integers(low=0, high=256, size=(64, 64), dtype=np.uint8)
    y = dct.block_dct2(dct.jpegio_to_jpeglib(x))
    qt = np.ones((1, 1, 8, 8), dtype='uint8')

    # extract foo features
    if feature_type is Features.FEATURES_JRM:
        features = jrm.extract(y, calibrated=False)
    elif feature_type is Features.FEATURES_CCJRM:
        features = jrm.extract(y, qt=qt, calibrated=True)
    elif feature_type is Features.FEATURES_GFR:
        features = gfr.extract(x)
    elif feature_type is Features.FEATURES_PHARM:
        features = pharm.extract(x)
    elif feature_type is Features.FEATURES_SPAM:
        features = spam.extract(x)
    elif feature_type is Features.FEATURES_SRM:
        features = srm.extract(x)
    elif feature_type is Features.FEATURES_SRMQ1:
        features = srm.srmq1.extract(x)
    elif feature_type is Features.FEATURES_CRM:
        x = rng.integers(low=0, high=256, size=(64, 64, 3), dtype=np.uint8)
        features = srm.crm.extract(x)
    else:
        raise ValueError("Unknown feature type")
    #
    return features


def group(
    features: np.ndarray,
    feature_type: Features,
) -> OrderedDict:
    """Split a feature vector into groups.

    :param features: feature vector
        of shape [num_features]
    :type features:
    :param feature_type: type of features
    :type feature_type:
    :return: ordered dict, where the keys are the submodel names
    :rtype:

    :Example:

    >>> # TODO
    """
    assert len(features.shape) == 1

    # Extract features from a dummy image. This gives us the submodel names and associated dimensionality for slicing the given input features.
    dummy_groups = generate_groups(feature_type)

    # Iterate over submodels
    groups = OrderedDict()
    next_col = 0
    for submodel, dummy_features in dummy_groups.items():
        num_features = len(dummy_features.flatten())

        # Take next slice from the input features and assign the submodel name
        groups[submodel] = features[..., next_col:next_col + num_features].reshape(dummy_features.shape)
        next_col += num_features

    assert next_col == len(features), "Mismatch in the total number of features"

    return groups


def flatten(features):
    """Flatten the submodels of a single sample to an ndarray.

    :param features: ordered dict, where the keys represent the submodel names and the values are the submodels
    :type features:
    :return: 1D array of length [num_features]

    :Example:

    >>> # TODO
    """
    return np.concatenate([
        f.flatten()
        for f in features.values()
    ])


# def group_batch(features, feature_type):
#     """
#     Split a flat feature vector into groups
#     :param features: ndarray of shape [num_samples, num_features]
#     :param feature_type: type of features
#     :return: ordered dict, where the keys are the submodel names
#     """

#     num_samples, total_num_features = features.shape

#     # Extract features from a dummy image. This gives us the submodel names and associated dimensionality for slicing the given input features.
#     dummy_groups = generate_groups(feature_type)

#     # Groups to return
#     groups = OrderedDict()

#     # Iterate over submodels
#     next_col = 0
#     for submodel_name, dummy_features in dummy_groups.items():
#         num_features = len(dummy_features.flatten())

#         # Take next slice from the input features and assign the submodel name
#         # Reshape to [num_samples, submodel_shape]
#         group_shape = (num_samples,) + dummy_features.shape
#         groups[submodel_name] = features[..., next_col:next_col + num_features].reshape(group_shape)

#         next_col += num_features

#     assert next_col == total_num_features, "Mismatch in the total number of features"

#     return groups


# def flatten_batch(batch_grouped_features):
#     """
#     Flatten a batch of grouped features to an ndarray.
#     :param batch_grouped_features: ordered dict, where the keys represent the submodel names and the values are the submodels, with one row per sample.
#     :return: 2D array of shape [num_samples, num_features]
#     """

#     # Batch size
#     num_samples = None
#     dtype = None

#     # Determine the total number of features
#     num_features = 0
#     for submodel_features_batch in batch_grouped_features.values():
#         submodel_num_samples = submodel_features_batch.shape[0]

#         # Verify that the number of samples matches the other submodels
#         if num_samples is None:
#             num_samples = submodel_num_samples
#             dtype = submodel_features_batch.dtype
#         else:
#             assert num_samples == submodel_num_samples, "Mismatch in the number of samples"
#             assert dtype == submodel_features_batch.dtype, "Mismatch in dtype"

#         num_features += np.prod(submodel_features_batch.shape[1:])

#     # Allocate output array
#     features = np.zeros((num_samples, num_features), dtype=dtype)

#     # Set up column pointer
#     next_col = 0

#     for submodel_features_batch in batch_grouped_features.values():
#         # Calculate number of features in current group
#         submodel_num_features = np.prod(submodel_features_batch.shape[1:])

#         # Copy into output array
#         features[:, next_col:next_col + submodel_num_features] = submodel_features_batch.reshape(num_samples, submodel_num_features)

#         # Advance column pointer
#         next_col += submodel_num_features

#     return features
