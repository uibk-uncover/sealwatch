import unittest
import numpy as np
from parameterized import parameterized
from sealwatch.features.jrm import extract_cc_jrm_features_from_file
from sealwatch.features.pharm.pharm_revisited import extract_pharm_revisited_features_from_file, PharmRevisitedFeatureExtractor
from sealwatch.utils.grouping import group_batch, flatten_batch, group_single, flatten_single
from sealwatch.utils.constants import CC_JRM, PHARM_REVISITED
import jpeglib
import tempfile


class TestGrouping(unittest.TestCase):

    @staticmethod
    def match_dicts(a, b):
        a_keys = list(a.keys())
        b_keys = list(b.keys())

        # Verify number of keys and order
        if a_keys != b_keys:
            return False

        for key in a_keys:
            # Verify values
            if not np.allclose(a[key], b[key]):
                return False

        return True

    @parameterized.expand(range(5))
    def test_ungroup_regroup_single_cc_jrm(self, seed):
        rng = np.random.RandomState(seed)
        img = rng.randint(low=0, high=256, size=(64, 64, 1), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".jpeg") as f:
            jpeglib.from_spatial(img).write_spatial(f.name, qt=75)
            features_grouped = extract_cc_jrm_features_from_file(f.name)

        features_flat = flatten_single(features_grouped)
        features_regrouped = group_single(features_flat, feature_type=CC_JRM)

        self.assertTrue(self.match_dicts(features_grouped, features_regrouped))

    @parameterized.expand(range(5))
    def test_ungroup_regroup_single_pharm(self, seed):
        rng = np.random.RandomState(seed)
        img = rng.randint(low=0, high=256, size=(64, 64, 1), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".jpeg") as f:
            qf = 75
            jpeglib.from_spatial(img).write_spatial(f.name, qt=qf)
            features_grouped = extract_pharm_revisited_features_from_file(f.name, q=PharmRevisitedFeatureExtractor.qf_to_quantization_step(qf))

        features_flat = flatten_single(features_grouped)
        features_regrouped = group_single(features_flat, feature_type=PHARM_REVISITED)

        self.assertTrue(self.match_dicts(features_grouped, features_regrouped))

    @parameterized.expand(range(5))
    def test_group_flatten_batch_cc_jrm(self, seed):
        rng = np.random.RandomState(seed)

        features = rng.rand(10, 22510)

        # Group
        features_grouped = group_batch(features, feature_type=CC_JRM)

        # Ungroup
        features_ungrouped = flatten_batch(features_grouped)

        # Compare
        np.testing.assert_array_equal(features, features_ungrouped)

    @parameterized.expand(range(5))
    def test_group_flatten_batch_pharm(self, seed):
        rng = np.random.RandomState(seed)

        # The number of features is num_projections * num_filters * T
        features = rng.rand(10, 7 * 2 * 100)

        # Group
        features_grouped = group_batch(features, feature_type=PHARM_REVISITED)

        # Ungroup
        features_ungrouped = flatten_batch(features_grouped)

        # Compare
        np.testing.assert_array_equal(features, features_ungrouped)


__all__ = ["TestGrouping"]
