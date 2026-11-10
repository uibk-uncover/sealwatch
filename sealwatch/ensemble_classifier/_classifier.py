"""Scikit-learn-style FLD ensemble classifier."""

from __future__ import annotations

import typing

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ._implementation import Implementation


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

    Advantages of FLD base learners:

    * Low training complexity.
    * A single FLD classifier is relatively weak and unstable, but forming an
      ensemble of them increases diversity.

    The decision threshold of each base learner is adjusted to minimize the total
    detector error under equal priors on the training set,
    :math:`P_E = \\min_{P_{FA}} \\frac{1}{2} (P_{FA} + P_{MD}(P_{FA}))`,
    where :math:`P_{FA}` is the probability of false alarms and :math:`P_{MD}` the
    probability of missed detections.

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
        All implementations give the same output.
    :param L: number of base learners; should be an integer. ``None`` determines it
        automatically from the OOB error.
    :param d_sub: subspace dimensionality; should be an integer. ``None`` finds it
        via a search over the OOB error (Algorithm 2 of the paper).
    :param search_L: maximum number of base learners used to evaluate each ``d_sub``
        candidate during the search. ``None`` trains every candidate to convergence,
        which reproduces the reference result exactly. A smaller integer (e.g., 50)
        speeds up the search at the price of a noisier choice of ``d_sub`` and
        a possibly slightly larger detection error. The selected ``d_sub`` is always
        trained to convergence. Ignored when ``d_sub`` is given.
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
    :param n_jobs: reserved for parallel training. Currently, training is always sequential.
    :param verbose: value 1 prints log messages; value 0 hides them.

    :ivar classes_: the two class labels, ``classes_[0]`` is cover and ``classes_[1]`` is stego.
    :ivar n_features_in_: number of features seen during :meth:`fit`.
    :ivar d_sub_: subspace dimensionality used by the trained ensemble.
    :ivar base_learners_: the trained base learners.
    :ivar training_records_: one record per training step (``d_sub``, number of
        base learners, OOB error), including the ``d_sub`` search.
    :ivar seed_subspaces_: seed used for sampling subspaces.
    :ivar seed_bootstrap_: seed used for sampling bootstraps.
    """

    def __init__(
        self,
        implementation: Implementation = Implementation.FLD_OPTIMIZED,
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> FldEnsembleClassifier:
        """Train an ensemble of Fisher linear discriminant classifiers.

        Training is considerably faster when ``X`` is C-contiguous; a warning is
        logged otherwise.

        :param X: training features of shape ``[num_samples, num_features]``, covers and stegos stacked.
        :param y: labels of shape ``[num_samples]`` with exactly two distinct values.
            Both classes must have the same number of samples and the same dimensionality.
        :return: the fitted classifier.
        :raises ValueError: if there are not exactly two classes, or if the classes
            differ in size.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by unweighted majority vote.

        Ties (equal votes for both classes) are resolved randomly with a fixed
        seed, as in the reference Matlab implementation.

        :param X: features of shape ``[num_samples, num_features]``.
        :return: labels of shape ``[num_samples]``, elements of :attr:`classes_`.
        :raises sklearn.exceptions.NotFittedError: if the classifier is not fitted.
        :raises ValueError: if the number of features differs from :meth:`fit`.
        """
        raise NotImplementedError

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """Alias of :meth:`decision_function`, kept for backward compatibility."""
        return self.decision_function(X)

    @property
    def num_base_learners(self) -> int:
        """Number of trained base learners."""
        return len(self.base_learners_)
