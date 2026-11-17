"""Unit tests of :class:`sealwatch.ensemble_classifier._oob.OutOfBagErrorEstimates`."""

import unittest

import numpy as np

from sealwatch.ensemble_classifier._oob import OutOfBagErrorEstimates


class TestOutOfBagErrorEstimates(unittest.TestCase):

    def test_initial_state(self):
        oob = OutOfBagErrorEstimates(num_covers=4, num_stegos=4)
        self.assertEqual(len(oob), 0)
        self.assertEqual(oob.errors, [])
        self.assertIsNone(oob.error)

    def test_error_without_ties(self):
        """Hand-computed: 3 covers and 3 stegos, all decided."""
        oob = OutOfBagErrorEstimates(num_covers=3, num_stegos=3)
        idx = np.array([0, 1, 2])
        # Covers: +, -, - -> one false alarm. Stegos: +, +, - -> one missed detection.
        error = oob.update(idx, np.array([2.0, -1.0, -0.5]), np.array([0.3, 5.0, -1.0]))
        self.assertAlmostEqual(error, 2 / 6)
        self.assertEqual(oob.errors, [error])
        self.assertEqual(oob.error, error)
        self.assertEqual(len(oob), 1)

    def test_votes_accumulate_over_learners(self):
        """A second learner can outvote the first on a row both tested."""
        oob = OutOfBagErrorEstimates(num_covers=1, num_stegos=1)
        oob.update(np.array([0]), np.array([1.0]), np.array([-1.0]))      # wrong, wrong: error 1
        self.assertAlmostEqual(oob.error, 1.0)
        oob.update(np.array([0]), np.array([-1.0]), np.array([1.0]))      # tie 0 + 0: random
        oob.update(np.array([0]), np.array([-1.0]), np.array([1.0]))      # now -1 / +1: right
        self.assertAlmostEqual(oob.error, 0.0)

    def test_only_oob_rows_are_updated(self):
        """Rows outside ``oob_indices`` keep their (undecided) state."""
        oob = OutOfBagErrorEstimates(num_covers=4, num_stegos=4)
        oob.update(np.array([1]), np.array([-1.0]), np.array([1.0]))
        np.testing.assert_array_equal(oob._votes_cover, [0, -1, 0, 0])
        np.testing.assert_array_equal(oob._votes_stego, [0, 1, 0, 0])

    def test_ties_use_seeded_rng_in_fixed_order(self):
        """Undecided rows are drawn from RandomState(1): covers first, then stegos."""
        oob = OutOfBagErrorEstimates(num_covers=5, num_stegos=3)
        error = oob.update(np.array([], dtype=int), np.array([]), np.array([]))  # all undecided

        rng = np.random.RandomState(1)
        false_alarms = np.sum(rng.rand(5) > 0.5)
        missed = np.sum(rng.rand(3) < 0.5)
        self.assertAlmostEqual(error, (false_alarms + missed) / 8)

    def test_reproducible(self):
        def run():
            oob = OutOfBagErrorEstimates(num_covers=20, num_stegos=20)
            rng = np.random.RandomState(0)
            for _ in range(10):
                idx = np.sort(rng.choice(20, size=7, replace=False))
                oob.update(idx, rng.randn(7), rng.randn(7))
            return oob.errors

        self.assertEqual(run(), run())


__all__ = ["TestOutOfBagErrorEstimates"]
