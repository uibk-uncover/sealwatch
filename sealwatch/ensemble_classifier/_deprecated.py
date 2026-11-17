"""Backward compatibility with earlier releases.

Thin wrappers around :class:`~sealwatch.ensemble_classifier.FldEnsembleClassifier`
that keep the old interface: separate cover and stego features, labels -1 (cover)
and +1 (stego), and a ``train()`` method returning the ensemble and its records.
They contain no logic of their own, and warn when used.
"""

import typing
import warnings

import numpy as np

from ._classifier import FldEnsembleClassifier
from ._implementation import Implementation


class _BaseLearnerView(object):
    """Read-only view of one base learner, with the attributes of the old ``BaseLearner``.

    :ivar subspace: indices of the feature dimensions of this base learner.
    :ivar learner: the trained :class:`~sealwatch.ensemble_classifier.FisherLinearDiscriminant`,
        with the weights ``w`` and the bias ``b``.
    """

    def __init__(self, subspace, learner):
        self.subspace = subspace
        self.learner = learner

    def predict(self, X: np.ndarray, subset: typing.Optional[np.ndarray] = None) -> np.ndarray:
        """Predict scores on the subspace, optionally only for the given rows."""
        X = X[:, self.subspace] if subset is None else X[subset][:, self.subspace]
        return self.learner.predict(X)


class EnsembleClassifier(object):
    """Deprecated. Use :class:`~sealwatch.ensemble_classifier.FldEnsembleClassifier`.

    Wraps a fitted classifier trained with the labels -1 (cover) and +1 (stego).
    Obtain it from :meth:`FldEnsembleTrainer.train`.

    :param classifier: fitted :class:`FldEnsembleClassifier` with ``classes_`` of -1 and +1.
    """

    def __init__(self, classifier: FldEnsembleClassifier):
        warnings.warn(
            "EnsembleClassifier is deprecated, use FldEnsembleClassifier",
            DeprecationWarning,
            stacklevel=2,
        )
        self._classifier = classifier

    @property
    def d_sub(self) -> int:
        """Subspace dimensionality of the trained ensemble."""
        return self._classifier.d_sub_

    @property
    def num_base_learners(self) -> int:
        """Number of trained base learners."""
        return self._classifier.num_base_learners

    @property
    def base_learners(self) -> typing.List[_BaseLearnerView]:
        """The trained base learners, each with the attributes ``subspace`` and ``learner``."""
        return [_BaseLearnerView(s, l) for s, l in zip(self._classifier.subspaces_, self._classifier.learners_)]

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """Calculate the confidence score based on majority voting.

        :param X: samples of shape ``[num_samples, num_features]``.
        :return: confidence of shape ``[num_samples]``, in the range of -1 for the
            negative and +1 for the positive class.
        """
        return self._classifier.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Calculate predictions based on (unweighted) majority voting. Ties are resolved randomly.

        :param X: samples of shape ``[num_samples, num_features]``.
        :return: predictions of shape ``[num_samples]``, -1 for the negative and +1 for the positive class.
        """
        return self._classifier.predict(X)

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate the accuracy.

        :param X: samples of shape ``[num_samples, num_features]``.
        :param y_true: labels of shape ``[num_samples]``, -1 for covers and +1 for stegos.
        :return: accuracy.
        :raises ValueError: if ``y_true`` contains other labels than -1 and +1.
        """
        if not set(np.unique(y_true)) <= {-1, +1}:
            raise ValueError("Expected input labels with values -1 and +1")
        return (y_true == self.predict(X)).mean()


class FldEnsembleTrainer(object):
    """Deprecated. Use :class:`~sealwatch.ensemble_classifier.FldEnsembleClassifier`.

    Trainer for creating an ensemble classifier consisting of Fisher linear
    discriminant base learners. This implementation assumes pairs of cover and
    stego images. Hence, the number of samples in ``Xc`` and ``Xs`` must be identical.
    See :class:`FldEnsembleClassifier` for how to choose ``d_sub`` and ``L``, and
    for the reference. Trains with ``FLD_LEGACY``, which reproduces the results of earlier releases exactly.

    :param Xc: cover training samples of shape ``[num_samples, num_features]``.
    :param Xs: stego training samples of shape ``[num_samples, num_features]``;
        should have the same number of samples and dimensionality as the covers.
    :param seed: seed for the main random number generator (``random_state`` of the classifier).
    :param seed_subspaces: seed for the subspace random number generator.
    :param seed_bootstrap: seed for the bootstrap random number generator.
    :param L: number of base learners; should be an integer; use ``"automatic"`` to determine this number automatically.
    :param d_sub: subspace dimensionality; should be an integer; use ``"automatic"`` to find this number via a search.
    :param verbose: value 1 prints log messages; value 0 hides them.
    :param max_num_base_learners: maximum number of base learners.
    :raises ValueError: if the features of covers and stegos differ in shape,
        or ``L`` or ``d_sub`` is neither an integer nor ``"automatic"``.
    """

    def __init__(
        self,
        Xc: np.ndarray,
        Xs: np.ndarray,
        seed: typing.Optional[int] = None,
        seed_subspaces: typing.Optional[int] = None,
        seed_bootstrap: typing.Optional[int] = None,
        L: typing.Union[int, str] = "automatic",
        d_sub: typing.Union[int, str] = "automatic",
        verbose: int = 1,
        max_num_base_learners: int = 500,
    ):
        warnings.warn(
            "FldEnsembleTrainer is deprecated, use FldEnsembleClassifier",
            DeprecationWarning,
            stacklevel=2,
        )
        if Xc.shape != Xs.shape:
            raise ValueError("Expected cover and stego features of the same shape")
        for name, value in (("L", L), ("d_sub", d_sub)):
            if not (isinstance(value, (int, np.integer)) or value == "automatic"):
                raise ValueError(f'Expected {name} to be an integer or the keyword "automatic"')

        self._Xc = Xc
        self._Xs = Xs
        self._classifier = FldEnsembleClassifier(
            implementation=Implementation.FLD_LEGACY,  # reproduces the results of earlier releases
            L=None if isinstance(L, str) else L,
            d_sub=None if isinstance(d_sub, str) else d_sub,
            max_num_base_learners=max_num_base_learners,
            random_state=seed,
            seed_subspaces=seed_subspaces,
            seed_bootstrap=seed_bootstrap,
            verbose=verbose,
        )

    def train(self) -> typing.Tuple[EnsembleClassifier, typing.List[dict]]:
        """Train an ensemble of Fisher linear discriminant classifiers.

        :return: ``(ensemble_classifier, training_records)`` as 2-tuple.
            ``ensemble_classifier`` is an :class:`EnsembleClassifier`;
            ``training_records`` is a list of dicts.
        """
        X = np.concatenate([self._Xc, self._Xs], axis=0)
        y = np.concatenate([-np.ones(len(self._Xc), dtype=int), +np.ones(len(self._Xs), dtype=int)])
        self._classifier.fit(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # already warned by the trainer
            ensemble = EnsembleClassifier(self._classifier)
        return ensemble, self._classifier.training_records_
