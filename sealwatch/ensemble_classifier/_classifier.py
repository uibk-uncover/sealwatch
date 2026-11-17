"""Scikit-learn-style FLD ensemble classifier."""

from __future__ import annotations

import os
import typing
import warnings

import numpy as np
from scipy.signal import convolve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .. import tools
from ._implementation import Implementation
from ._oob import OutOfBagErrorEstimates
from ._search import SubspaceDimensionalitySearch
from .fld import FisherLinearDiscriminant


log = tools.setup_custom_logger(os.path.basename(__file__))

# Moving-average kernel of the stopping criterion: averages 5 consecutive OOB estimates
_L_KERNEL = np.ones(5) / 5


def _select(
    X: np.ndarray,
    rows: typing.Optional[np.ndarray] = None,
    cols: typing.Optional[np.ndarray] = None,
) -> np.ndarray:
    """Take rows and columns of a 2D array, same as ``X[rows, :][:, cols]`` but faster.

    The rows may repeat (bootstrap samples). See
    https://stackoverflow.com/questions/14386822/fast-numpy-fancy-indexing

    :param X: array of shape ``[num_samples, num_features]``.
    :param rows: row indices, or ``None`` for all rows.
    :param cols: column indices, or ``None`` for all columns.
    :return: the selected block.
    """
    if rows is not None and cols is not None:
        return X.ravel()[(cols + (rows * X.shape[1]).reshape((-1, 1))).ravel()].reshape(rows.size, cols.size)
    if cols is not None:
        return X[:, cols]
    if rows is not None:
        return X[rows]
    return X


def _split_classes(X: np.ndarray, y: np.ndarray, neg, pos, dtype: np.dtype) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Split ``X`` into covers and stegos along ``y``.

    If each class is a single contiguous block of rows, and ``X`` already has the
    working dtype and is C-contiguous, zero-copy views are returned (the data are only
    read during training), which halves the peak memory. Otherwise, a contiguous copy
    of each class is made, which also handles arbitrary label orders and dtype changes.

    :param X: features of shape ``[num_samples, num_features]``.
    :param y: labels of shape ``[num_samples]``.
    :param neg: label of the covers.
    :param pos: label of the stegos.
    :param dtype: target dtype.
    :return: ``(Xc, Xs)``.
    """
    def block_slice(mask):
        idx = np.flatnonzero(mask)
        # Contiguous iff the indices form a gap-free run
        if len(idx) and idx[-1] - idx[0] + 1 == len(idx):
            return slice(int(idx[0]), int(idx[-1]) + 1)
        return None

    sl_c = block_slice(y == neg)
    sl_s = block_slice(y == pos)
    if sl_c is not None and sl_s is not None and X.dtype == dtype and X.flags["C_CONTIGUOUS"]:
        # Row blocks of a C-contiguous array are C-contiguous views: no data is copied
        return X[sl_c], X[sl_s]

    return (np.ascontiguousarray(X[y == neg], dtype=dtype),
            np.ascontiguousarray(X[y == pos], dtype=dtype))


class FldEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble of Fisher linear discriminant (FLD) base learners for binary steganalysis.

    Introduced in

        J. Kodovský, J. Fridrich, V. Holub.
        Ensemble classifiers for steganalysis of digital media.
        IEEE Transactions on Information Forensics and Security, vol. 7, 2012.

    Each base learner is an FLD trained on a random feature subspace of
    dimensionality ``d_sub`` and a bootstrap sample of the training data. Predictions
    are made by unweighted majority vote. This single class is both trainer and
    fitted model: call :meth:`fit`, then :meth:`predict` or :meth:`score`.

    **Paired data.** The implementation assumes pairs of cover and stego images.
    Both classes must have the same number of samples and the same dimensionality,
    and the i-th sample of each class belongs to the same cover image (the
    bootstrap and OOB bookkeeping shares row indices between both classes).

    **Labels.** Any two labels are accepted. The smaller label (``classes_[0]``) is
    the negative class (cover), the larger (``classes_[1]``) the positive class
    (stego). :meth:`predict` returns elements of ``classes_``.

    **Setting the subspace dimensionality** ``d_sub``

    * If ``d_sub`` is set too low, the individual base learners will not be able to learn anything.
    * If ``d_sub`` is set too high, the individual base learners will become more
      dependent, which decreases the chance of learning non-linear classification
      boundaries. Furthermore, individual base learners may overfit when the number
      of training examples is relatively low compared to the number of feature dimensions.

    **Setting the number of base learners** ``L``

    * The ensemble benefits from more base learners until performance starts to saturate.
    * For automatic selection of ``L``, the progress of the OOB estimate is observed.
      Training terminates when the last ``L_memory`` (default 50) moving averages,
      each calculated from 5 consecutive OOB estimates, lie in an epsilon-tube.

    Example::

        clf = sw.ensemble_classifier.FldEnsembleClassifier(random_state=12345)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = clf.decision_function(X_test)

    :param implementation: code path to use, see :class:`~sealwatch.ensemble_classifier.Implementation`.
        ``FLD_LEGACY`` reproduces the legacy implementation exactly for the same seed.
    :param L: number of base learners; should be an integer. ``None`` determines it
        automatically from the OOB error.
    :param d_sub: subspace dimensionality; should be an integer. ``None`` finds it
        via a search over the OOB error (Algorithm 2 of the paper).
    :param search_L: maximum number of base learners used to evaluate each ``d_sub``
        candidate during the search. ``None`` trains every candidate to convergence,
        as the reference implementation does. A smaller integer (e.g., 50)
        speeds up the search at the price of a noisier choice of ``d_sub`` and
        a possibly slightly larger detection error. The selected ``d_sub`` is always
        trained to convergence. Ignored when ``d_sub`` or ``L`` is given.
    :param max_num_base_learners: maximum number of base learners.
    :param oob_error_tolerance: tolerance :math:`\\tau` of the ``d_sub`` search.
    :param initial_d_sub_step: initial step :math:`\\delta_d` of the ``d_sub`` search.
    :param L_min_length: minimum number of base learners before the automatic stopping applies.
    :param L_memory: number of moving averages inspected by the stopping criterion.
    :param L_epsilon: width of the epsilon-tube of the stopping criterion.
    :param dtype: floating-point type of the feature matrices. ``None`` selects float64.
        float32 halves the memory of the (large) feature matrices and speeds up
        the BLAS operations, but it biases the OOB error estimates that drive
        model selection. With the automatic ``d_sub`` search, the search then picks a
        too small ``d_sub`` and the detection error degrades (about 2 percentage
        points on real data). Use float32 only together with a fixed ``d_sub``.
    :param random_state: seed for the main random number generator, from which
        ``seed_subspaces`` and ``seed_bootstrap`` are derived.
    :param seed_subspaces: seed for the subspace random number generator.
        Overrides the value derived from ``random_state``.
    :param seed_bootstrap: seed for the bootstrap random number generator.
        Overrides the value derived from ``random_state``.
    :param n_jobs: number of base learners trained in parallel. Not implemented yet:
        training is sequential, and other values than ``None`` and 1 give a :class:`FutureWarning`.
    :param verbose: value 1 prints log messages; value 0 hides them.

    :ivar classes_: the two class labels, ``classes_[0]`` is cover and ``classes_[1]`` is stego.
    :ivar n_features_in_: number of features seen during :meth:`fit`.
    :ivar d_sub_: subspace dimensionality used by the trained ensemble.
    :ivar subspaces_: feature indices of each base learner, a list of arrays of length ``d_sub_``.
    :ivar learners_: the trained base learners, a list of :class:`~sealwatch.ensemble_classifier.FisherLinearDiscriminant`.
    :ivar training_records_: one record per training run (``d_sub``, ``num_base_learners``,
        ``oob_error``). With ``search_L``, the last record is the final training of the selected ``d_sub``.
    :ivar seed_subspaces_: seed used for sampling subspaces.
    :ivar seed_bootstrap_: seed used for sampling bootstraps.
    """

    def __init__(
        self,
        implementation: Implementation = Implementation.FLD_LEGACY,
        L: typing.Optional[int] = None,
        d_sub: typing.Optional[int] = None,
        search_L: typing.Optional[int] = None,
        max_num_base_learners: int = 500,
        oob_error_tolerance: float = 0.02,
        initial_d_sub_step: int = 200,
        L_min_length: int = 25,
        L_memory: int = 50,
        L_epsilon: float = 0.005,
        dtype: typing.Optional[np.dtype] = None,
        random_state: typing.Optional[int] = None,
        seed_subspaces: typing.Optional[int] = None,
        seed_bootstrap: typing.Optional[int] = None,
        n_jobs: typing.Optional[int] = None,
        verbose: int = 0,
    ):
        # scikit-learn contract: only store the arguments, do no work here
        self.implementation = implementation
        self.L = L
        self.d_sub = d_sub
        self.search_L = search_L
        self.max_num_base_learners = max_num_base_learners
        self.oob_error_tolerance = oob_error_tolerance
        self.initial_d_sub_step = initial_d_sub_step
        self.L_min_length = L_min_length
        self.L_memory = L_memory
        self.L_epsilon = L_epsilon
        self.dtype = dtype
        self.random_state = random_state
        self.seed_subspaces = seed_subspaces
        self.seed_bootstrap = seed_bootstrap
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _generate_random_subspace(self, rng_subspaces: np.random.RandomState, d_sub: int, max_dim: int) -> np.ndarray:
        """Draw a random subspace of ``d_sub`` out of ``max_dim`` features.

        A permutation is built on top of ``rand()``, which yields the same values as
        Matlab's Mersenne twister, so that the subspaces match the reference implementations.
        """
        # Drawing the subset directly, rng_subspaces.choice(max_dim, size=d_sub, replace=False),
        # is faster but gives other subspaces for the same seed, and so other models.
        # We tried it: the detection error is statistically the same, and the gain matters
        # only for tiny training sets, where the draw of all max_dim values dominates.
        return tools.matlab.randperm_naive(rng_subspaces, max_dim)[:d_sub]

    @staticmethod
    def _regenerate_bootstrap_samples(rng_bootstrap: np.random.RandomState, n: int) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Draw a bootstrap sample.

        :param rng_bootstrap: random number generator.
        :param n: number of samples per class.
        :return: ``(train_indices, test_indices)``. The training indices are ``n`` draws
            with replacement (may contain duplicates). The test indices are the
            out-of-bag rows, i.e., those never drawn, in ascending order.
        """
        train_indices = np.floor(n * rng_bootstrap.rand(n)).astype(int)
        test_indices = np.where(np.bincount(train_indices, minlength=n) == 0)[0]
        return train_indices, test_indices

    def _has_next_base_learner(self, i: int, oob: OutOfBagErrorEstimates, max_L: int) -> bool:
        """Decide whether to add another base learner (Eq. 6 of the paper).

        :param i: number of base learners trained so far.
        :param oob: OOB error estimates with the history so far.
        :param max_L: maximum number of base learners.
        :return: ``True`` if another base learner should be added.
        """
        # Fixed L
        if self.L is not None:
            return i < self.L

        # Automatic L: stop when the moving average of the OOB error lies in an epsilon-tube
        if len(oob) == 0:
            return False
        if i >= max_L:
            if self.verbose:
                log.info("Maximum number of base learners reached")
            return False
        if len(oob) < self.L_min_length:
            return True
        errors = oob.errors
        A = convolve(errors[max(0, len(errors) - self.L_memory):], _L_KERNEL, mode="valid")
        V = np.abs(np.max(A) - np.min(A))  # Eq. 6 / 7
        return V >= self.L_epsilon

    def _train_ensemble(
        self,
        Xc: np.ndarray,
        Xs: np.ndarray,
        d_sub: int,
        max_L: int,
        rng_subspaces: np.random.RandomState,
        rng_bootstrap: np.random.RandomState,
        keep_learners: bool,
    ):
        """Train an ensemble for a fixed ``d_sub``.

        Base learners are added until the stopping criterion fires or ``max_L`` is reached.

        :param Xc: cover features of shape ``[n, num_features]``.
        :param Xs: stego features of shape ``[n, num_features]``.
        :param d_sub: subspace dimensionality.
        :param max_L: maximum number of base learners.
        :param rng_subspaces: random number generator for subspaces.
        :param rng_bootstrap: random number generator for bootstraps.
        :param keep_learners: if ``False``, the base learners are discarded after
            their OOB votes are counted (to save memory when only the error matters).
        :return: ``(subspaces, learners, oob_error, num_base_learners)``. The first two
            are lists, or ``None`` if the learners are not kept.
        """
        n, max_dim = Xc.shape
        oob = OutOfBagErrorEstimates(num_covers=n, num_stegos=n)
        subspaces = [] if keep_learners else None
        learners = [] if keep_learners else None
        num_base_learners = 0

        has_next = True
        while has_next:
            subspace = self._generate_random_subspace(rng_subspaces, d_sub, max_dim)
            train_idx, test_idx = self._regenerate_bootstrap_samples(rng_bootstrap, n)

            # Train on the bootstrap sample projected on the subspace
            learner = FisherLinearDiscriminant(implementation=self.implementation)
            learner.fit_presplit(
                _select(Xc, rows=train_idx, cols=subspace),
                _select(Xs, rows=train_idx, cols=subspace))
            num_base_learners += 1
            if keep_learners:
                subspaces.append(subspace)
                learners.append(learner)

            # Test on the out-of-bag samples
            oob.update(
                test_idx,
                learner.predict(_select(Xc, rows=test_idx, cols=subspace)),
                learner.predict(_select(Xs, rows=test_idx, cols=subspace)))
            if self.verbose:
                log.info(f"  - d_sub {d_sub}, OOB {oob.error:.4f}, L {num_base_learners}")

            has_next = self._has_next_base_learner(num_base_learners, oob, max_L)

        return subspaces, learners, oob.error, num_base_learners

    def fit(self, X: np.ndarray, y: np.ndarray) -> FldEnsembleClassifier:
        """Train an ensemble of Fisher linear discriminant classifiers.

        Training is considerably faster when ``X`` is C-contiguous.

        :param X: training features of shape ``[num_samples, num_features]``, covers and stegos stacked.
        :param y: labels of shape ``[num_samples]`` with exactly two distinct values.
            Both classes must have the same number of samples and the same dimensionality.
        :return: the fitted classifier.
        :raises ValueError: if there are not exactly two classes, if the classes
            differ in size, if ``implementation`` is unknown, or if ``d_sub`` exceeds the
            number of features.
        """
        if not isinstance(self.implementation, Implementation):
            raise ValueError(f"Unknown implementation: {self.implementation!r}")
        if self.n_jobs not in (None, 1):
            warnings.warn(
                f"n_jobs={self.n_jobs} is ignored: parallel training is not implemented yet, "
                "and will come in a future release; training is sequential",
                FutureWarning,
                stacklevel=2,
            )
        X, y = check_X_y(X, y, dtype=None, order="C")
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("FldEnsembleClassifier supports exactly two classes")
        neg, pos = self.classes_[0], self.classes_[1]
        self.n_features_in_ = X.shape[1]

        dtype = np.dtype(self.dtype) if self.dtype is not None else np.dtype(np.float64)
        Xc, Xs = _split_classes(X, y, neg, pos, dtype)
        if Xc.shape[0] != Xs.shape[0]:
            raise ValueError(
                f"Expected equal cover/stego counts (got {Xc.shape[0]} and {Xs.shape[0]}); "
                "this classifier assumes paired data.")
        n, max_dim = Xc.shape
        if self.d_sub is not None and self.d_sub > max_dim:
            raise ValueError(f"d_sub ({self.d_sub}) exceeds the number of features ({max_dim})")

        # Random number generators. Derive the seeds in the same order as the reference.
        rng_main = np.random.RandomState(self.random_state)
        seed_subspaces = self.seed_subspaces
        if not seed_subspaces:
            seed_subspaces = int(np.ceil(rng_main.random() * 1e9))
        seed_bootstrap = self.seed_bootstrap
        if not seed_bootstrap:
            seed_bootstrap = int(np.ceil(rng_main.random() * 1e9))
        self.seed_subspaces_ = seed_subspaces
        self.seed_bootstrap_ = seed_bootstrap
        rngs = dict(
            rng_subspaces=np.random.RandomState(seed_subspaces),
            rng_bootstrap=np.random.RandomState(seed_bootstrap),
        )

        records = []

        def record(d_sub, oob_error, num_base_learners):
            records.append({"d_sub": d_sub, "oob_error": oob_error, "num_base_learners": num_base_learners})

        if self.d_sub is None and max_dim > 1:
            # Algorithm 2: search for d_sub
            search = SubspaceDimensionalitySearch(
                d_sub_step=self.initial_d_sub_step,
                max_dim=max_dim,
                oob_error_tolerance=self.oob_error_tolerance)

            if self.search_L is None or self.L is not None:
                # Train every candidate to convergence and keep the best ensemble
                min_oob_error = None
                while search.search_in_progress:
                    d_sub = search.next_d_sub
                    subspaces, learners, oob_error, num_base_learners = self._train_ensemble(
                        Xc, Xs, d_sub, self.max_num_base_learners, keep_learners=True, **rngs)
                    if min_oob_error is None or oob_error < min_oob_error:
                        min_oob_error = oob_error
                        optimal_d_sub, best = d_sub, (subspaces, learners)
                    search.update(d_sub, oob_error)
                    record(d_sub, oob_error, num_base_learners)
                    if self.verbose:
                        log.info("")
                self.subspaces_, self.learners_ = best

            else:
                # Rank the candidates with at most search_L base learners, discard them,
                # and train only the winner to convergence
                while search.search_in_progress:
                    d_sub = search.next_d_sub
                    _, _, oob_error, num_base_learners = self._train_ensemble(
                        Xc, Xs, d_sub, self.search_L, keep_learners=False, **rngs)
                    search.update(d_sub, oob_error)
                    record(d_sub, oob_error, num_base_learners)
                    if self.verbose:
                        log.info("")
                optimal_d_sub = int(search.optimal_d_sub)
                self.subspaces_, self.learners_, oob_error, num_base_learners = self._train_ensemble(
                    Xc, Xs, optimal_d_sub, self.max_num_base_learners, keep_learners=True, **rngs)
                record(optimal_d_sub, oob_error, num_base_learners)

        else:
            # Fixed d_sub
            optimal_d_sub = 1 if max_dim == 1 else self.d_sub
            self.subspaces_, self.learners_, oob_error, num_base_learners = self._train_ensemble(
                Xc, Xs, optimal_d_sub, self.max_num_base_learners, keep_learners=True, **rngs)
            record(optimal_d_sub, oob_error, num_base_learners)

        self.d_sub_ = optimal_d_sub
        self.training_records_ = records
        self._vote_weights_ = None  # built on the first prediction
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _validate_for_prediction(self, X: np.ndarray) -> np.ndarray:
        """Check that the classifier is fitted and ``X`` has the right shape.

        :raises sklearn.exceptions.NotFittedError: if the classifier is not fitted.
        :raises ValueError: if the number of features differs from :meth:`fit`.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=None)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but FldEnsembleClassifier was "
                f"fitted with {self.n_features_in_} features.")
        return X

    def _get_vote_weights(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Build (and cache) the weights of the whole ensemble as one matrix.

        Column ``i`` is the weight vector of base learner ``i`` scattered back into the
        full feature space (zero outside its subspace). So ``X @ W - b`` yields the
        scores of all base learners in one matrix product. Built on the first prediction,
        so that the memory is only paid when the model is used for inference.

        :return: ``(W, b)`` of shapes ``[num_features, L]`` and ``[L]``, in the dtype of the weights.
        """
        if self._vote_weights_ is None:
            L = len(self.learners_)
            dtype = np.asarray(self.learners_[0].w).dtype
            W = np.zeros((self.n_features_in_, L), dtype=dtype)
            b = np.empty(L, dtype=dtype)
            for i, (subspace, learner) in enumerate(zip(self.subspaces_, self.learners_)):
                W[subspace, i] = learner.w
                b[i] = learner.b
            self._vote_weights_ = (W, b)
        return self._vote_weights_

    def _vote(self, X: np.ndarray) -> np.ndarray:
        """Sum of the signed votes of all base learners; positive means stego.

        :param X: validated features of shape ``[num_samples, num_features]``.
        :return: integer vote sums of shape ``[num_samples]``.
        """
        W, b = self._get_vote_weights()
        return np.sign(X @ W - b).sum(axis=1).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the confidence of each sample being of the positive class.

        The confidence is the fraction of base learners voting for ``classes_[1]``
        minus the fraction voting for ``classes_[0]``. Positive values indicate
        ``classes_[1]`` (stego).

        :param X: features of shape ``[num_samples, num_features]``.
        :return: confidences of shape ``[num_samples]`` in the range ``[-1, 1]``.
        :raises sklearn.exceptions.NotFittedError: if the classifier is not fitted.
        :raises ValueError: if the number of features differs from :meth:`fit`.
        """
        return self._vote(self._validate_for_prediction(X)) / len(self.learners_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by unweighted majority vote.

        Ties (equal votes for both classes) are resolved randomly with a fixed
        seed, as in the reference Matlab implementation.

        :param X: features of shape ``[num_samples, num_features]``.
        :return: labels of shape ``[num_samples]``, elements of :attr:`classes_`.
        :raises sklearn.exceptions.NotFittedError: if the classifier is not fitted.
        :raises ValueError: if the number of features differs from :meth:`fit`.
        """
        votes = self._vote(self._validate_for_prediction(X))
        rng_for_ties = np.random.RandomState(6020)
        tie_mask = votes == 0
        votes[tie_mask] = np.sign(rng_for_ties.rand(int(np.sum(tie_mask))) - 0.5).astype(int)
        return np.where(votes > 0, self.classes_[1], self.classes_[0])

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """Alias of :meth:`decision_function`, kept for backward compatibility."""
        return self.decision_function(X)

    @property
    def num_base_learners(self) -> int:
        """Number of trained base learners."""
        return len(self.learners_)
