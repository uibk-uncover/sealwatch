"""Tests of :mod:`sealwatch.ensemble_classifier`.

Groups
------
1. Reference: the training trajectory of ``FLD_LEGACY`` matches Matlab.
2. Equivalence: ``FLD_LEGACY`` reproduces the legacy package bit for bit (frozen outputs in
   ``assets/ensemble/numpy/reference.npz``, made with the ``ensemble_classifier`` of release 2026.11
   by ``make_reference.py`` in the same folder).
3. Labels: any two labels work.
4. Scikit-learn contract and input validation.
5. Parameters: fixed ``L``, fixed ``d_sub``, ``search_L``, ``dtype``.
6. Robustness: degenerate data.
7. Deprecated :class:`FldEnsembleTrainer` and :class:`EnsembleClassifier`.

Trainings on the Matlab tutorial data are expensive and shared between tests
through :func:`_fit_tutorial`.

Run from the repository root (asset paths are relative).
"""

import functools
import unittest

import numpy as np
import scipy.io
from parameterized import parameterized
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

import sealwatch as sw

from . import defs


TUTORIAL = {
    12345: "ensemble/matlab/tutorial_seed_12345.mat",
    98765: "ensemble/matlab/tutorial_seed_98765.mat",
}
FldEnsembleClassifier = sw.ensemble_classifier.FldEnsembleClassifier
FisherLinearDiscriminant = sw.ensemble_classifier.FisherLinearDiscriminant


@functools.lru_cache(maxsize=None)
def _legacy_reference():
    """Outputs of the legacy package (release 2026.11), see the module docstring."""
    with np.load(defs.ASSETS_DIR / "ensemble/numpy/reference.npz") as f:
        return dict(f)


@functools.lru_cache(maxsize=None)
def _load_tutorial(seed):
    """Load the Matlab tutorial data and its reference training trajectory.

    :return: ``(X, y, X_test, y_test, mat)``, labels in {-1, +1}.
    """
    mat = scipy.io.loadmat(defs.ASSETS_DIR / TUTORIAL[seed])
    Xc = np.ascontiguousarray(mat["TRN_cover"])
    Xs = np.ascontiguousarray(mat["TRN_stego"])
    Xc_test, Xs_test = mat["TST_cover"], mat["TST_stego"]
    X = np.concatenate([Xc, Xs])
    y = np.concatenate([-np.ones(len(Xc), dtype=int), np.ones(len(Xs), dtype=int)])
    X_test = np.concatenate([Xc_test, Xs_test])
    y_test = np.concatenate([-np.ones(len(Xc_test), dtype=int), np.ones(len(Xs_test), dtype=int)])
    return X, y, X_test, y_test, mat


@functools.lru_cache(maxsize=None)
def _fit_tutorial(seed, implementation=sw.FLD_LEGACY, search_L=None):
    """Fit once per configuration; do not mutate the returned classifier."""
    X, y, *_ = _load_tutorial(seed)
    return FldEnsembleClassifier(
        implementation=implementation, search_L=search_L, random_state=seed).fit(X, y)


def _make_paired(n, d, n_signal=8, shift=0.35, seed=0):
    """Synthetic paired cover/stego features with a weak signal.

    :return: ``(X, y)``, covers first, labels in {-1, +1}.
    """
    rng = np.random.RandomState(seed)
    Xc = rng.randn(n, d)
    Xs = Xc + rng.randn(n, d) * 0.1
    Xs[:, rng.choice(d, size=min(n_signal, d), replace=False)] += shift
    X = np.ascontiguousarray(np.concatenate([Xc, Xs]))
    y = np.concatenate([-np.ones(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def _detection_error(y_true, y_pred):
    """:math:`P_E = (P_{FA} + P_{MD}) / 2` under equal priors, labels in {-1, +1}."""
    p_fa = np.mean(y_pred[y_true == -1] == 1)
    p_md = np.mean(y_pred[y_true == 1] == -1)
    return 0.5 * (p_fa + p_md)


def _threshold_search_loop(y_true, num_covers, num_stegos):
    """Reference: the original loop over the sorted samples (0 cover, 1 stego)."""
    num_samples = num_covers + num_stegos
    MD, FA = 0, num_covers                  # case 1: covers score lower
    MD2, FA2 = num_stegos, 0                # case 2: covers score higher
    E_min = FA + MD
    threshold_idx = sgn = None
    for idx in range(num_samples - 1):
        if y_true[idx] == 0:
            FA, MD2 = FA - 1, MD2 + 1
        else:
            MD, FA2 = MD + 1, FA2 - 1
        if FA + MD < E_min:
            E_min, threshold_idx, sgn = FA + MD, idx, 1
        if FA2 + MD2 < E_min:
            E_min, threshold_idx, sgn = FA2 + MD2, idx, -1
    return threshold_idx, sgn


class TestReference(unittest.TestCase):
    """1. The training trajectory (d_sub, L, OOB error per step) of FLD_LEGACY matches Matlab."""

    @parameterized.expand([(seed,) for seed in TUTORIAL])
    def test_matlab_trajectory(self, seed):
        clf = _fit_tutorial(seed)
        mat = _load_tutorial(seed)[-1]

        d_sub = mat["search_d_sub"].flatten()
        L = mat["search_L"].flatten()
        oob = mat["search_oob"].flatten()
        self.assertEqual(len(clf.training_records_), len(d_sub))
        for step, record in enumerate(clf.training_records_):
            self.assertEqual(record["d_sub"], int(d_sub[step]), f"d_sub, step {step}")
            self.assertEqual(record["num_base_learners"], int(L[step]), f"L, step {step}")
            self.assertTrue(np.isclose(record["oob_error"], oob[step]), f"OOB, step {step}")


class TestEquivalence(unittest.TestCase):
    """2. The output is the same as of the legacy package, not only the trajectory."""

    def test_fld_matches_legacy_package(self):
        """The FLD reproduces the original FLD bit for bit, including the threshold
        tie-breaking, on random data with many duplicate scores."""
        ref = _legacy_reference()
        rng = np.random.RandomState(0)  # the generator of the reference trials
        for trial in range(200):
            n = rng.randint(2, 60)
            X = np.round(rng.randn(2 * n, 5), rng.randint(0, 2))  # coarse -> ties
            y = np.concatenate([-np.ones(n, dtype=int), np.ones(n, dtype=int)])
            X[n:, 0] += rng.rand()
            fld = FisherLinearDiscriminant().fit(X, (y + 1) // 2)
            np.testing.assert_array_equal(fld.w, ref["fld_w"][trial], err_msg=f"trial {trial}")
            self.assertEqual(fld.b, ref["fld_b"][trial], msg=f"trial {trial}")

    def test_fld_threshold_search_is_identical(self):
        """The vectorized search selects exactly the same threshold as the original loop."""
        fld = FisherLinearDiscriminant
        rng = np.random.RandomState(1)
        for trial in range(500):
            nc, ns = rng.randint(1, 40, size=2)
            y_true = rng.permutation(np.concatenate([np.zeros(nc), np.ones(ns)]))
            self.assertEqual(
                _threshold_search_loop(y_true, nc, ns),
                fld._search_threshold(y_true, nc, ns),
                f"trial {trial}")

    @parameterized.expand([(seed,) for seed in TUTORIAL])
    def test_legacy_matches_legacy_package_exactly(self, seed):
        """FLD_LEGACY reproduces the original implementation bit for bit."""
        prefix = f"tutorial_{seed}_"
        ref = {k[len(prefix):]: v for k, v in _legacy_reference().items() if k.startswith(prefix)}
        X_test = _load_tutorial(seed)[2]
        new = _fit_tutorial(seed)

        records = [[r["d_sub"], r["num_base_learners"], r["oob_error"]] for r in new.training_records_]
        np.testing.assert_array_equal(records, ref["records"])  # d_sub, L, and the OOB errors
        np.testing.assert_array_equal(new.subspaces_, ref["subspaces"])
        np.testing.assert_array_equal([b.w for b in new.learners_], ref["w"])
        np.testing.assert_array_equal([b.b for b in new.learners_], ref["b"])
        np.testing.assert_array_equal(new.decision_function(X_test), ref["confidence"])
        np.testing.assert_array_equal(new.predict(X_test), ref["predict"])

    def test_fld_validation(self):
        fld = FisherLinearDiscriminant
        X = np.random.RandomState(2).randn(10, 3)
        with self.assertRaises(ValueError):
            fld().fit(X, np.ones(10, dtype=int))                    # one class only
        with self.assertRaises(ValueError):
            fld(implementation="fast").fit_presplit(X[:5], X[5:])   # unknown implementation
        with self.assertRaises(AttributeError):
            fld().predict(X)                                        # not fitted

    def test_reproducible_and_deterministic_ties(self):
        """Same seed gives the same model, another seed another; ties in ``predict``
        are broken with a fixed seed (L=2 produces many)."""
        X, y = _make_paired(100, 40, seed=1)
        a = FldEnsembleClassifier(d_sub=20, L=2, random_state=1).fit(X, y)
        b = FldEnsembleClassifier(d_sub=20, L=2, random_state=1).fit(X, y)
        c = FldEnsembleClassifier(d_sub=20, L=2, random_state=2).fit(X, y)
        np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))
        self.assertFalse(np.array_equal(a.decision_function(X), c.decision_function(X)))
        np.testing.assert_array_equal(a.predict(X), a.predict(X))


class TestLabels(unittest.TestCase):
    """3. Any two labels are accepted; the smaller one is the cover."""

    @parameterized.expand([
        ("zero_one", 0, 1),
        ("strings", "cover", "stego"),   # sorts: "cover" < "stego"
    ])
    def test_encoding_does_not_change_decisions(self, _name, neg, pos):
        X, y = _make_paired(100, 40, seed=3)
        reference = FldEnsembleClassifier(d_sub=20, L=15, random_state=3).fit(X, y)

        clf = FldEnsembleClassifier(d_sub=20, L=15, random_state=3).fit(X, np.where(y == -1, neg, pos))

        self.assertEqual(list(clf.classes_), [neg, pos])
        np.testing.assert_array_equal(clf.decision_function(X), reference.decision_function(X))
        np.testing.assert_array_equal(clf.predict(X), np.where(reference.predict(X) == -1, neg, pos))

    def test_decision_function_range_and_sign(self):
        X, y = _make_paired(100, 40, seed=4)
        clf = FldEnsembleClassifier(d_sub=20, L=21, random_state=4).fit(X, y)
        conf = clf.decision_function(X)
        self.assertTrue(np.all(np.abs(conf) <= 1.0))
        non_tie = conf != 0
        np.testing.assert_array_equal(
            clf.predict(X)[non_tie], np.where(conf[non_tie] > 0, clf.classes_[1], clf.classes_[0]))


class TestContractAndValidation(unittest.TestCase):
    """4. Scikit-learn interface and input validation."""

    def test_clone_and_params(self):
        clf = FldEnsembleClassifier(d_sub=10, search_L=40, random_state=1)
        copy = clone(clf)
        self.assertEqual(copy.get_params(), clf.get_params())
        copy.set_params(d_sub=20)
        self.assertEqual((copy.d_sub, clf.d_sub), (20, 10))
        self.assertFalse(hasattr(clf, "learners_"))  # constructor does no work

    def test_fit_and_fitted_attributes(self):
        X, y = _make_paired(80, 30, seed=6)
        clf = FldEnsembleClassifier(d_sub=15, L=10, random_state=6)
        self.assertIs(clf.fit(X, y), clf)
        self.assertEqual(clf.n_features_in_, 30)
        self.assertEqual(clf.d_sub_, 15)
        self.assertEqual(clf.num_base_learners, 10)
        self.assertEqual(len(clf.learners_), 10)
        for subspace in clf.subspaces_:
            self.assertEqual(len(np.unique(subspace)), 15)

    def test_not_fitted(self):
        clf = FldEnsembleClassifier()
        for method in (clf.predict, clf.decision_function):
            with self.assertRaises(NotFittedError):
                method(np.zeros((3, 5)))

    def test_wrong_number_of_features_at_predict(self):
        X, y = _make_paired(60, 20, seed=9)
        clf = FldEnsembleClassifier(d_sub=10, L=5, random_state=9).fit(X, y)
        with self.assertRaises(ValueError):
            clf.predict(X[:, :10])

    def test_invalid_input_rejected(self):
        X, y = _make_paired(40, 10, seed=10)
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        cases = {
            "single class": (X, np.ones(len(X), dtype=int)),
            "three classes": (X[:30], np.repeat([0, 1, 2], 10)),
            "unpaired sizes": (X[:35], np.concatenate([-np.ones(15, dtype=int), np.ones(20, dtype=int)])),
            "non-finite": (X_nan, y),
        }
        for name, (X_bad, y_bad) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                FldEnsembleClassifier(d_sub=5, L=3).fit(X_bad, y_bad)

    def test_invalid_implementation_rejected(self):
        X, y = _make_paired(40, 10, seed=14)
        with self.assertRaises((ValueError, TypeError)):
            FldEnsembleClassifier(implementation="fast", d_sub=5, L=3).fit(X, y)

    def test_n_jobs_warns(self):
        """Parallel training is not implemented; asking for it warns, and trains sequentially."""
        X, y = _make_paired(40, 10, seed=16)
        reference = FldEnsembleClassifier(d_sub=5, L=3, random_state=16).fit(X, y)
        with self.assertWarns(FutureWarning):
            clf = FldEnsembleClassifier(d_sub=5, L=3, n_jobs=4, random_state=16).fit(X, y)
        np.testing.assert_array_equal(clf.decision_function(X), reference.decision_function(X))

    def test_input_not_modified(self):
        X, y = _make_paired(60, 20, seed=15)
        X_before, y_before = X.copy(), y.copy()
        FldEnsembleClassifier(d_sub=10, L=5, random_state=15).fit(X, y)
        np.testing.assert_array_equal(X, X_before)
        np.testing.assert_array_equal(y, y_before)

    def test_exports(self):
        self.assertIsInstance(sw.FLD_LEGACY, sw.ensemble_classifier.Implementation)
        self.assertIs(FldEnsembleClassifier().implementation, sw.FLD_LEGACY)  # the default


class TestParameters(unittest.TestCase):
    """5. Fixed L / d_sub, search_L, dtype."""

    def test_fixed_d_sub_and_L(self):
        X, y = _make_paired(100, 64, seed=20)
        clf = FldEnsembleClassifier(d_sub=32, L=30, random_state=20).fit(X, y)
        self.assertEqual((clf.d_sub_, clf.num_base_learners), (32, 30))

    def test_fixed_L_automatic_d_sub(self):
        X, y = _make_paired(100, 64, seed=21)
        clf = FldEnsembleClassifier(L=20, random_state=21).fit(X, y)
        self.assertEqual(clf.num_base_learners, 20)
        self.assertTrue(1 <= clf.d_sub_ <= 64)

    def test_fixed_d_sub_automatic_L(self):
        X, y = _make_paired(100, 64, seed=22)
        clf = FldEnsembleClassifier(d_sub=32, random_state=22).fit(X, y)
        self.assertEqual(clf.d_sub_, 32)
        self.assertGreaterEqual(clf.num_base_learners, clf.L_min_length)

    def test_max_num_base_learners_respected(self):
        X, y = _make_paired(100, 64, seed=23)
        clf = FldEnsembleClassifier(d_sub=32, max_num_base_learners=12, random_state=23).fit(X, y)
        self.assertLessEqual(clf.num_base_learners, 12)

    def test_search_L_caps_candidates(self):
        X, y = _make_paired(200, 128, seed=25)
        clf = FldEnsembleClassifier(search_L=10, random_state=25).fit(X, y)
        search = clf.training_records_[:-1]  # all but the final converged training
        self.assertGreaterEqual(len(search), 1)
        self.assertLessEqual(max(r["num_base_learners"] for r in search), 10)

    def test_search_L_ignored_with_fixed_d_sub(self):
        X, y = _make_paired(100, 64, seed=26)
        a = FldEnsembleClassifier(d_sub=32, L=10, random_state=26).fit(X, y)
        b = FldEnsembleClassifier(d_sub=32, L=10, search_L=5, random_state=26).fit(X, y)
        np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))

    @parameterized.expand([(seed,) for seed in TUTORIAL])
    def test_search_L_close_to_exact(self, seed):
        """A capped search costs at most a small increase in detection error."""
        _, _, X_test, y_test, _ = _load_tutorial(seed)
        exact = _fit_tutorial(seed)
        capped = _fit_tutorial(seed, search_L=50)
        pe_exact = _detection_error(y_test, exact.predict(X_test))
        pe_capped = _detection_error(y_test, capped.predict(X_test))
        self.assertLessEqual(pe_capped, pe_exact + 0.02)

    def test_float32_singular_retry(self):
        """A singular float32 scatter matrix is regularized with the float32 epsilon. The float64
        epsilon vanishes next to values of order 1 in float32, which made the retry loop spin."""
        sigma = np.ones((2, 2), dtype=np.float32)  # singular
        w = FisherLinearDiscriminant()._solve(sigma, np.array([1.0, -1.0], dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(w)))
        eps = np.finfo(np.float32).eps
        np.testing.assert_array_equal(sigma, [[1 + eps, 1], [1, 1 + eps]])  # one retry, in place

    def test_float32(self):
        X, y = _make_paired(100, 48, seed=27)
        X32 = X.astype(np.float32)
        clf = FldEnsembleClassifier(d_sub=24, L=10, dtype=np.float32, random_state=27).fit(X32, y)
        self.assertEqual(clf.predict(X32).shape, (len(X),))


class TestRobustness(unittest.TestCase):
    """6. Degenerate data must neither crash nor produce non-finite scores."""

    def test_constant_features(self):
        """Zero-variance columns make the scatter matrix singular."""
        X, y = _make_paired(100, 40, seed=30)
        X[:, :10] = 1.0
        clf = FldEnsembleClassifier(d_sub=20, L=10, random_state=30).fit(X, y)
        self.assertTrue(np.all(np.isfinite(clf.decision_function(X))))

    def test_more_features_than_samples(self):
        X, y = _make_paired(20, 100, seed=33)
        clf = FldEnsembleClassifier(d_sub=30, L=5, random_state=33).fit(X, y)
        self.assertTrue(np.all(np.isfinite(clf.decision_function(X))))


class TestDeprecatedTrainer(unittest.TestCase):
    """7. The old trainer interface still works, with a warning."""

    def test_trainer(self):
        _, _, _, _, mat = _load_tutorial(12345)
        with self.assertWarns(DeprecationWarning):
            trainer = sw.ensemble_classifier.FldEnsembleTrainer(
                Xc=np.ascontiguousarray(mat["TRN_cover"]),
                Xs=np.ascontiguousarray(mat["TRN_stego"]), seed=12345, verbose=0)
        ensemble, records = trainer.train()

        # Matlab trajectory
        np.testing.assert_array_equal([r["d_sub"] for r in records], mat["search_d_sub"].flatten())
        np.testing.assert_array_equal(
            [r["num_base_learners"] for r in records], mat["search_L"].flatten())
        # Same model as the FLD_LEGACY classifier, with the old -1/+1 interface
        clf = _fit_tutorial(12345)  # the compatibility layer reproduces earlier releases
        X_test = _load_tutorial(12345)[2]
        np.testing.assert_array_equal(ensemble.predict(X_test), clf.predict(X_test))
        np.testing.assert_allclose(ensemble.predict_confidence(X_test), clf.decision_function(X_test))
        self.assertTrue(set(np.unique(ensemble.predict(X_test))) <= {-1, 1})

    def test_ensemble_classifier(self):
        """The old EnsembleClassifier wraps a fitted classifier with labels -1 and +1."""
        X, y = _make_paired(60, 20, seed=40)
        clf = FldEnsembleClassifier(d_sub=10, L=7, random_state=40).fit(X, y)
        with self.assertWarns(DeprecationWarning):
            ensemble = sw.ensemble_classifier.EnsembleClassifier(clf)

        self.assertEqual((ensemble.d_sub, ensemble.num_base_learners), (10, 7))
        base_learner = ensemble.base_learners[0]
        np.testing.assert_array_equal(base_learner.subspace, clf.subspaces_[0])
        np.testing.assert_array_equal(
            base_learner.predict(X, subset=np.arange(5)), clf.learners_[0].predict(X[:5, clf.subspaces_[0]]))
        np.testing.assert_array_equal(ensemble.predict_confidence(X), clf.decision_function(X))
        np.testing.assert_array_equal(ensemble.predict(X), clf.predict(X))
        self.assertEqual(ensemble.score(X, y), np.mean(clf.predict(X) == y))
        with self.assertRaises(ValueError):
            ensemble.score(X, (y + 1) // 2)  # labels 0 and 1


__all__ = [
    "TestReference", "TestEquivalence", "TestLabels", "TestContractAndValidation",
    "TestParameters", "TestRobustness", "TestDeprecatedTrainer",
]
