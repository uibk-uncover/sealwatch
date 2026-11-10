"""Regression tests for the output-equivalent FLD ensemble re-implementation
:mod:`sealwatch.ensemble_classifier_rework2`.

Unlike :mod:`sealwatch.ensemble_classifier_rework`, which is bit-identical to the
Matlab reference, ``rework2`` trades bit-identity for training speed (a capped-L
``d_sub`` search, motivated by the flat ``d_sub`` minimum and fast ``L`` saturation
in Kodovsky et al., 2012, Fig. 2). It is therefore validated on **output
equivalence** rather than exact reproduction: the detection error ``P_E`` on a
held-out test set must match the legacy baseline within a tolerance, together with
the usual scikit-learn contract and input edge cases.
"""

import importlib
import unittest

import numpy as np
import scipy.io

from . import defs


LEGACY = "sealwatch.ensemble_classifier"
REWORK2 = "sealwatch.ensemble_classifier_rework2"


def _make_paired(n, d, n_signal=8, shift=0.35, seed=0):
    """Build synthetic paired cover/stego features with a weak signal.

    :param n: number of samples per class.
    :param d: feature dimensionality.
    :param n_signal: number of feature dimensions carrying the (weak) stego signal.
    :param shift: mean shift applied to the stego signal dimensions.
    :param seed: RNG seed.
    :return: ``(X, y)`` with ``X`` of shape ``[2 n, d]`` (float64, C-contiguous) and
        labels ``y`` in ``{-1, +1}`` (covers first, stegos second).
    """
    rng = np.random.RandomState(seed)
    Xc = rng.randn(n, d)
    Xs = Xc + rng.randn(n, d) * 0.1
    Xs[:, rng.choice(d, size=min(n_signal, d), replace=False)] += shift
    X = np.ascontiguousarray(np.concatenate([Xc, Xs]).astype(np.float64))
    y = np.concatenate([-np.ones(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def _detection_error(clf, cover, stego):
    """Detection error ``P_E = 0.5 (P_FA + P_MD)`` under equal priors (paper Eq. 1)."""
    p_fa = np.mean(clf.predict(cover) == +1)   # cover classified as stego
    p_md = np.mean(clf.predict(stego) == -1)   # stego classified as cover
    return 0.5 * (p_fa + p_md)


class TestFldEnsembleClassifierRework2(unittest.TestCase):
    """Output-equivalence and interface tests for ``rework2``."""

    # Maximum tolerated increase in P_E over the legacy baseline. Comfortably above
    # the run-to-run noise of the capped search (~5e-3) yet small enough to catch a
    # real degradation (e.g. the float32 search bias, which costs ~2e-2).
    PE_TOLERANCE = 0.02

    def setUp(self):
        self.legacy = importlib.import_module(LEGACY)
        self.rework2 = importlib.import_module(REWORK2)

    def test_security_matches_legacy_baseline(self):
        """``P_E`` on the held-out Matlab tutorial test set is within tolerance of
        the legacy full-search trainer -- the core "match the outputs" criterion."""
        mat = scipy.io.loadmat(
            defs.ASSETS_DIR / "ensemble_matlab/tutorial_seed_12345.mat")
        Xc = np.ascontiguousarray(mat["TRN_cover"])
        Xs = np.ascontiguousarray(mat["TRN_stego"])
        cover_tst, stego_tst = mat["TST_cover"], mat["TST_stego"]
        X = np.concatenate([Xc, Xs])
        y = np.concatenate([-np.ones(len(Xc), dtype=int), np.ones(len(Xs), dtype=int)])

        legacy_ensemble, _ = self.legacy.FldEnsembleTrainer(
            Xc=Xc, Xs=Xs, seed=12345, verbose=0).train()
        pe_legacy = _detection_error(legacy_ensemble, cover_tst, stego_tst)

        clf = self.rework2.FldEnsembleClassifier(
            random_state=12345, verbose=0).fit(X, y)
        pe_rework2 = _detection_error(clf, cover_tst, stego_tst)

        self.assertLessEqual(
            pe_rework2, pe_legacy + self.PE_TOLERANCE,
            f"rework2 P_E {pe_rework2:.4f} exceeds legacy {pe_legacy:.4f} "
            f"by more than {self.PE_TOLERANCE}")

    def test_fixed_dsub_skips_search(self):
        """A fixed ``d_sub`` bypasses the compass search entirely (no search phase)."""
        X, y = _make_paired(200, 64, seed=1)
        clf = self.rework2.FldEnsembleClassifier(
            d_sub=32, L=30, random_state=1).fit(X, y)
        self.assertEqual(clf.d_sub_, 32)
        self.assertEqual(clf.num_base_learners, 30)
        phases = [record["phase"] for record in clf.training_records_]
        self.assertEqual(phases, ["final"])

    def test_search_effort_caps_candidate_cost(self):
        """A low ``search_effort`` ranks each ``d_sub`` candidate with at most the
        derived learner cap, i.e. the speed knob actually limits the search work."""
        X, y = _make_paired(200, 128, seed=2)
        clf = self.rework2.FldEnsembleClassifier(
            search_effort=0.1, random_state=2).fit(X, y)
        cap = clf._search_cap()
        self.assertEqual(cap, 10)
        search_counts = [r["num_base_learners"]
                         for r in clf.training_records_ if r["phase"] == "search"]
        self.assertGreaterEqual(len(search_counts), 1)
        self.assertLessEqual(max(search_counts), cap)

    def test_sklearn_contract(self):
        """``clone`` works and predict/decision_function/score behave as documented."""
        from sklearn.base import clone
        X, y = _make_paired(150, 48, seed=3)
        estimator = self.rework2.FldEnsembleClassifier(d_sub=24, L=20, random_state=3)
        clone(estimator)  # exercises get_params/set_params; must not raise
        estimator.fit(X, y)
        self.assertEqual(estimator.predict(X).shape, (len(X),))
        confidence = estimator.decision_function(X)
        self.assertTrue(np.all(confidence >= -1.0) and np.all(confidence <= 1.0))
        self.assertTrue(0.0 <= estimator.score(X, y) <= 1.0)

    def test_predict_matches_decision_function_sign(self):
        """Non-tie predictions agree with the sign of ``decision_function``."""
        X, y = _make_paired(150, 48, seed=7)
        clf = self.rework2.FldEnsembleClassifier(d_sub=24, L=21, random_state=7).fit(X, y)
        confidence = clf.decision_function(X)
        predictions = clf.predict(X)
        non_tie = confidence != 0
        expected = np.where(confidence[non_tie] > 0, clf.classes_[1], clf.classes_[0])
        np.testing.assert_array_equal(predictions[non_tie], expected)

    def test_requires_two_classes(self):
        """A single-class label vector is rejected."""
        X, _ = _make_paired(50, 16, seed=4)
        with self.assertRaises(ValueError):
            self.rework2.FldEnsembleClassifier(d_sub=8, L=5).fit(
                X, np.ones(len(X), dtype=int))

    def test_requires_paired_class_sizes(self):
        """Unequal cover/stego counts violate the paired-data assumption."""
        rng = np.random.RandomState(5)
        X = rng.randn(90, 16)
        y = np.concatenate([-np.ones(40, dtype=int), np.ones(50, dtype=int)])
        with self.assertRaises(ValueError):
            self.rework2.FldEnsembleClassifier(d_sub=8, L=5).fit(X, y)

    def test_float32_dtype_runs(self):
        """The float32 path trains and predicts (memory mode; see class docstring)."""
        X, y = _make_paired(150, 48, seed=6)
        clf = self.rework2.FldEnsembleClassifier(
            d_sub=24, L=20, dtype=np.float32, random_state=6)
        clf.fit(X.astype(np.float32), y)
        self.assertEqual(clf.predict(X.astype(np.float32)).shape, (len(X),))


__all__ = ["TestFldEnsembleClassifierRework2"]
