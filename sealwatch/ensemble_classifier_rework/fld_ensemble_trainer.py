import os
import numpy as np
from scipy.signal import convolve

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from .base_learner import BaseLearner
from .out_of_bag_error_estimates import OutOfBagErrorEstimates
from .subspace_dimensionality_search import (
    SubspaceDimensionalitySearch,
    FixedDimensionalityDummySearch,
)
from .. import tools


log = tools.setup_custom_logger(os.path.basename(__file__))


class FldEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble of Fisher linear discriminant (FLD) base learners for binary
    steganalysis, with a scikit-learn compatible interface.

    Re-implementation of the trainer from J. Kodovsky, J. Fridrich, V. Holub,
    "Ensemble Classifiers for Steganalysis of Digital Media", IEEE TIFS 7(2), 2012.
    This single class is both trainer and fitted model: call :meth:`fit`, then
    :meth:`predict` / :meth:`score`. With ``matlab_compat=True`` (default) the
    results are bit-identical to the legacy ``FldEnsembleTrainer``.

    The implementation assumes *paired* cover and stego samples; both classes must
    contain the same number of samples (the bootstrap/OOB bookkeeping shares row
    indices between both classes).

    :param L: number of base learners; ``None`` selects it automatically.
    :param d_sub: subspace dimensionality; ``None`` triggers the grid search.
    :param max_num_base_learners: hard cap on the number of base learners.
    :param oob_error_tolerance: tolerance tau for the d_sub grid search.
    :param initial_d_sub_step: initial step delta_d for the d_sub grid search.
    :param L_min_length: minimum number of base learners before automatic stopping.
    :param L_memory: number of moving averages inspected by the stopping criterion.
    :param L_epsilon: width of the epsilon-tube for the stopping criterion.
    :param random_state: seed for the main RNG (derives the two sub-seeds).
    :param seed_subspaces: explicit seed for the subspace RNG.
    :param seed_bootstrap: explicit seed for the bootstrap RNG.
    :param n_jobs: reserved for parallel training (added in a later step); currently
        training is always sequential.
    :param matlab_compat: if True, reproduce the legacy/Matlab behaviour exactly
        (float64, naive randperm). If False, use faster subspace sampling and the
        input dtype.
    :param dtype: working dtype for the feature matrices. None = float64 in
        matlab_compat mode, otherwise the input dtype. Set np.float32 to halve the
        memory of the (large) feature matrices and speed up the BLAS (~1.7x); the
        scatter/solve then run in float32 too (a dtype-aware ridge in fld.py keeps
        them stable). WARNING: float32 is only safe with a *fixed* d_sub. With the
        automatic d_sub search it biases the OOB error estimates that drive model
        selection, so the search picks a too-small d_sub and the detection error P_E
        degrades (~2 percentage points on real data); use float64 for the auto search.
    :param verbose: 1 prints progress, 0 stays quiet.
    """

    def __init__(
        self,
        L=None,
        d_sub=None,
        max_num_base_learners=500,
        oob_error_tolerance=0.02,
        initial_d_sub_step=200,
        L_min_length=25,
        L_memory=50,
        L_epsilon=0.005,
        random_state=None,
        seed_subspaces=None,
        seed_bootstrap=None,
        n_jobs=1,
        matlab_compat=True,
        dtype=None,
        verbose=0,
    ):
        # scikit-learn contract: only store arguments, do no work here.
        self.L = L
        self.d_sub = d_sub
        self.max_num_base_learners = max_num_base_learners
        self.oob_error_tolerance = oob_error_tolerance
        self.initial_d_sub_step = initial_d_sub_step
        self.L_min_length = L_min_length
        self.L_memory = L_memory
        self.L_epsilon = L_epsilon
        self.random_state = random_state
        self.seed_subspaces = seed_subspaces
        self.seed_bootstrap = seed_bootstrap
        self.n_jobs = n_jobs
        self.matlab_compat = matlab_compat
        self.dtype = dtype
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # RNG-consuming draws
    # ------------------------------------------------------------------ #
    def _generate_random_subspace(self, rng_subspaces, d_sub, max_dim):
        if self.matlab_compat:
            return tools.matlab.randperm_naive(rng_subspaces, max_dim)[:d_sub]
        return rng_subspaces.choice(max_dim, size=d_sub, replace=False)

    @staticmethod
    def _regenerate_bootstrap_samples(rng_bootstrap, n):
        # n draws with replacement (may contain duplicates)
        train_indices = np.floor(n * rng_bootstrap.rand(n)).astype(int)
        # Out-of-bag = never drawn. O(n) via bincount; same sorted order as the
        # legacy np.setdiff1d(np.arange(n), train_indices).
        test_indices = np.where(np.bincount(train_indices, minlength=n) == 0)[0]
        return train_indices, test_indices

    # ------------------------------------------------------------------ #
    # OOB bookkeeping (identical to the legacy _update_oob_error_estimates)
    # ------------------------------------------------------------------ #
    def _update_oob_error_estimates(self, base_learner, test_indices, oob, Xc, Xs):
        Xc_proj = base_learner.predict(Xc, subset=test_indices)
        Xs_proj = base_learner.predict(Xs, subset=test_indices)

        oob.Xc_num[test_indices] += 1
        oob.Xs_num[test_indices] += 1
        oob.Xc_fusion_majority_vote[test_indices] += np.sign(Xc_proj).astype(int)
        oob.Xs_fusion_majority_vote[test_indices] += np.sign(Xs_proj).astype(int)

        num_covers = len(oob.Xc_fusion_majority_vote)
        num_stegos = len(oob.Xs_fusion_majority_vote)

        num_fp = np.sum(oob.Xc_fusion_majority_vote > 0)
        num_undecided = np.sum(oob.Xc_fusion_majority_vote == 0)
        num_fp += np.sum(oob.rng_for_ties.rand(num_undecided) > 0.5)

        num_md = np.sum(oob.Xs_fusion_majority_vote < 0)
        num_undecided = np.sum(oob.Xs_fusion_majority_vote == 0)
        num_md += np.sum(oob.rng_for_ties.rand(num_undecided) < 0.5)

        oob.ys.append((num_fp + num_md) / (num_covers + num_stegos))
        return oob

    def _has_next_base_learner(self, i, oob):
        if self.L is None:
            if len(oob.ys) == 0:
                return False
            if len(oob.ys) < self.L_min_length:
                return True
            ys = oob.ys
            A = convolve(ys[max(0, len(ys) - self.L_memory):], self._L_kernel, mode="valid")
            V = np.abs(np.max(A) - np.min(A))  # Eq. 6 / 7
            if V < self.L_epsilon:
                return False
            if i == self.max_num_base_learners:
                if self.verbose:
                    log.info("Maximum number of base learners reached")
                return False
            return True
        return i < self.L

    @staticmethod
    def _split_classes(X, y, neg, pos, dtype):
        """
        Split X into (cover, stego) along y.

        Fast path: if each class occupies a single contiguous row block of X and X
        already has the working dtype and is C-contiguous, return zero-copy *views*
        (Xc/Xs are read-only during training). Otherwise fall back to a contiguous
        per-class copy, which also handles arbitrary label orders and dtype changes.
        """
        target = np.dtype(dtype)

        def block_slice(mask):
            idx = np.flatnonzero(mask)
            # contiguous iff the indices form a gap-free run
            if len(idx) and idx[-1] - idx[0] + 1 == len(idx):
                return slice(int(idx[0]), int(idx[-1]) + 1)
            return None

        sl_c = block_slice(y == neg)
        sl_s = block_slice(y == pos)
        if (sl_c is not None and sl_s is not None
                and X.dtype == target and X.flags["C_CONTIGUOUS"]):
            # Row-block slices of a C-contiguous array are themselves C-contiguous
            # views -- no data copied.
            return X[sl_c], X[sl_s]

        return (np.ascontiguousarray(X[y == neg], dtype=target),
                np.ascontiguousarray(X[y == pos], dtype=target))

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        """
        Train the ensemble.

        :param X: features of shape [num_samples, num_features].
        :param y: binary labels; the smaller class label is treated as cover (-1),
            the larger as stego (+1). Both classes must have the same size.
        :return: self
        """
        X, y = check_X_y(X, y, dtype=None, order="C")
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("FldEnsembleClassifier supports exactly two classes")
        neg, pos = self.classes_[0], self.classes_[1]

        # Number of input features (scikit-learn convention; checked in predict)
        self.n_features_in_ = X.shape[1]

        # In matlab_compat mode we must use float64. Otherwise keep a floating
        # input dtype (e.g. float32 for less memory), but never train on integers.
        if self.dtype is not None:
            dtype = np.dtype(self.dtype)
        elif self.matlab_compat:
            dtype = np.float64
        else:
            dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float32

        # Split into cover/stego. Xc and Xs are only ever *read* during training, so
        # when each class is a single contiguous block of X and no dtype conversion is
        # needed, take zero-copy *views* into X instead of duplicating the whole
        # dataset. This is the common case (the legacy shim hands over cover-then-stego
        # float64 data) and roughly halves the peak memory, since the boolean-indexed
        # copy below would otherwise hold a second full copy of the data for the entire
        # fit. Falls back to a contiguous copy for arbitrary label orders / dtypes.
        Xc, Xs = self._split_classes(X, y, neg, pos, dtype)
        if Xc.shape[0] != Xs.shape[0]:
            raise ValueError(
                f"Expected equal cover/stego counts (got {Xc.shape[0]} and "
                f"{Xs.shape[0]}); this trainer assumes paired data."
            )

        n = Xc.shape[0]
        max_dim = Xc.shape[1]

        # RNG setup -- replicates the legacy derivation order for bit-identity.
        rng_main = np.random.RandomState(self.random_state)
        seed_subspaces = self.seed_subspaces
        if not seed_subspaces:
            seed_subspaces = int(np.ceil(rng_main.random() * 1e9))
        seed_bootstrap = self.seed_bootstrap
        if not seed_bootstrap:
            seed_bootstrap = int(np.ceil(rng_main.random() * 1e9))
        self.seed_subspaces_ = seed_subspaces
        self.seed_bootstrap_ = seed_bootstrap
        rng_subspaces = np.random.RandomState(seed_subspaces)
        rng_bootstrap = np.random.RandomState(seed_bootstrap)

        if self.L is None:
            self._L_kernel = np.ones(5) / 5

        # d_sub search setup
        if self.d_sub is None and max_dim > 1:
            search = SubspaceDimensionalitySearch(
                d_sub_step=self.initial_d_sub_step,
                max_dim=max_dim,
                oob_error_tolerance=self.oob_error_tolerance,
            )
        else:
            search = FixedDimensionalityDummySearch(d_sub=1 if max_dim == 1 else self.d_sub)

        min_oob_error = 1.0
        optimal_d_sub = None
        trained_ensemble = None
        training_records = []

        # Outer loop: search over d_sub
        while search.search_in_progress:
            d_sub = search.next_d_sub
            num_base_learners = 0
            has_next = True
            base_learners = []
            oob = OutOfBagErrorEstimates(num_covers=n, num_stegos=n)

            # Inner loop: add base learners until the stopping criterion fires
            while has_next:
                subspace = self._generate_random_subspace(rng_subspaces, d_sub, max_dim)
                train_idx, test_idx = self._regenerate_bootstrap_samples(rng_bootstrap, n)

                base_learner = BaseLearner()
                base_learner.fit(Xc=Xc, Xs=Xs, subspace=subspace, subset=train_idx)
                base_learners.append(base_learner)
                num_base_learners += 1

                oob = self._update_oob_error_estimates(base_learner, test_idx, oob, Xc, Xs)
                if self.verbose:
                    log.info(f"  - d_sub {d_sub}, OOB {oob.ys[-1]:.4f}, L {num_base_learners}")

                has_next = self._has_next_base_learner(num_base_learners, oob)

            oob_error = oob.ys[-1]
            if oob_error < min_oob_error:
                min_oob_error = oob_error
                optimal_d_sub = d_sub
                trained_ensemble = base_learners

            search.update(d_sub, oob_error)
            training_records.append({
                "d_sub": d_sub,
                "oob_error": oob_error,
                "num_base_learners": num_base_learners,
            })
            if self.verbose:
                log.info("")
            del base_learners, oob

        self.base_learners_ = trained_ensemble
        self.d_sub_ = optimal_d_sub
        self.training_records_ = training_records
        self._vote_weights_ = None  # lazily built on first predict (see _vote)
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _validate_for_prediction(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=None)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but FldEnsembleClassifier was "
                f"fitted with {self.n_features_in_} features."
            )
        return X

    def _get_vote_weights(self):
        """
        Build (and cache) the dense ``(n_features, L)`` weight matrix and length-L
        bias vector for the whole ensemble. Column ``i`` is base learner ``i``'s
        weight vector scattered back into the full feature space (zeros outside its
        subspace), so ``X @ W`` reproduces every learner's projected score in one
        matmul. Built lazily on first predict so the memory is only paid when the
        model is actually used for inference.
        """
        cached = getattr(self, "_vote_weights_", None)
        if cached is None:
            L = len(self.base_learners_)
            # Match the learners' weight dtype (float32 model -> float32 W, half the
            # memory of the dense (n_features, L) matrix).
            dtype = np.asarray(self.base_learners_[0].learner.w).dtype
            W = np.zeros((self.n_features_in_, L), dtype=dtype)
            b = np.empty(L, dtype=dtype)
            for i, base_learner in enumerate(self.base_learners_):
                W[base_learner.subspace, i] = base_learner.learner.w
                b[i] = base_learner.learner.b
            cached = (W, b)
            self._vote_weights_ = cached
        return cached

    def _vote(self, X):
        # Single matmul over the entire ensemble instead of a Python loop of L
        # per-learner projections. Each base learner's score is
        #   X[:, subspace] @ w - b  ==  X @ W[:, i] - b[i]
        # (the scattered weight matrix is zero outside the subspace), so the signed
        # majority vote is reproduced exactly while running ~100x faster for large L.
        W, b = self._get_vote_weights()
        scores = X @ W - b
        return np.sign(scores).sum(axis=1).astype(int)

    def decision_function(self, X):
        """Confidence in [-1, +1] from the signed majority vote."""
        X = self._validate_for_prediction(X)
        return self._vote(X) / len(self.base_learners_)

    def predict(self, X):
        """Predict class labels. Ties are broken randomly (seeded, as in the legacy code)."""
        X = self._validate_for_prediction(X)
        votes = self._vote(X)
        rng_for_ties = np.random.RandomState(6020)
        tie_mask = votes == 0
        votes[tie_mask] = np.sign(rng_for_ties.rand(int(np.sum(tie_mask))) - 0.5).astype(int)
        y = np.sign(votes).astype(int)
        return self.classes_[(y + 1) // 2]

    def predict_confidence(self, X):
        return self.decision_function(X)

    @property
    def base_learners(self):
        return self.base_learners_

    @property
    def num_base_learners(self):
        return len(self.base_learners_)


class FldEnsembleTrainer(FldEnsembleClassifier):
    """
    Backward-compatible shim for the legacy interface: accepts ``Xc``/``Xs`` and
    exposes ``train()`` returning ``(self, training_records)``.
    """

    def __init__(self, Xc, Xs, seed=None, seed_subspaces=None, seed_bootstrap=None,
                 L="automatic", d_sub="automatic", verbose=1, max_num_base_learners=500):
        super().__init__(
            L=None if L == "automatic" else L,
            d_sub=None if d_sub == "automatic" else d_sub,
            max_num_base_learners=max_num_base_learners,
            random_state=seed,
            seed_subspaces=seed_subspaces,
            seed_bootstrap=seed_bootstrap,
            verbose=verbose,
        )
        self._Xc = Xc
        self._Xs = Xs

    def train(self):
        X = np.concatenate([self._Xc, self._Xs], axis=0)
        y = np.concatenate([-np.ones(len(self._Xc), dtype=int),
                            +np.ones(len(self._Xs), dtype=int)])
        self.fit(X, y)
        return self, self.training_records_
