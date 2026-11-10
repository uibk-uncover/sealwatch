"""Deprecated trainer interface of earlier releases."""

import typing
import warnings

import numpy as np

from ._classifier import FldEnsembleClassifier


class FldEnsembleTrainer(object):
    """Deprecated. Use :class:`~sealwatch.ensemble_classifier.FldEnsembleClassifier`.

    Trainer for creating an ensemble classifier consisting of Fisher linear
    discriminant base learners. This implementation assumes pairs of cover and
    stego images. Hence, the number of samples in ``Xc`` and ``Xs`` must be identical.
    See :class:`FldEnsembleClassifier` for how to choose ``d_sub`` and ``L``, and
    for the reference.

    Wraps :class:`FldEnsembleClassifier` with the interface of earlier releases:
    the cover and stego features are passed separately and :meth:`train` returns
    the trained ensemble together with the training records.

    For backward compatibility, the returned ensemble predicts with the labels
    -1 (cover) and +1 (stego), and its confidence is in the range of -1 for the
    negative and +1 for the positive class.

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
        raise NotImplementedError

    def train(self) -> typing.Tuple[FldEnsembleClassifier, typing.List[dict]]:
        """Train an ensemble of Fisher linear discriminant classifiers.

        :return: ``(ensemble_classifier, training_records)`` as 2-tuple.
            ``ensemble_classifier`` is a fitted :class:`FldEnsembleClassifier`;
            ``training_records`` is a list of dicts.
        """
        raise NotImplementedError
