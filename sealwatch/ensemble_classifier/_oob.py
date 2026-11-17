"""Out-of-bag (OOB) error estimates of a growing ensemble."""

from __future__ import annotations

import typing

import numpy as np


class OutOfBagErrorEstimates(object):
    """Running out-of-bag error of an ensemble that grows one base learner at a time.

    The name out-of-bag estimate comes from bagging (bootstrap aggregation), which is
    an established technique for reducing the variance of classifiers. Each base
    learner is trained on a bootstrap sample. The rows it has not seen (out-of-bag)
    act as a held-out test set for that learner. Summing the votes that each row
    receives from the learners for which it was out-of-bag, and taking the majority,
    estimates the detection error of the ensemble without a separate validation set
    (Eq. 5 of Kodovský et al., 2012).

    Usage: after training each base learner, call :meth:`update` with the learner's
    scores on its out-of-bag rows. The history of errors, one per base learner,
    is available as :attr:`errors`.

    Rows without a majority (equal votes, including rows never out-of-bag) are decided
    at random. The random number generator is seeded with a fixed value, so that
    the estimates are reproducible and match the Matlab implementation. The order
    and number of random draws are part of that contract.

    :param num_covers: number of cover samples.
    :param num_stegos: number of stego samples.
    """

    def __init__(self, num_covers: int, num_stegos: int):
        self.num_covers = num_covers
        self.num_stegos = num_stegos

        # While adding base learners to the ensemble, keep track of the votes:
        # per sample, the sum of the +1/-1 votes of the learners that tested it.
        self._votes_cover = np.zeros(num_covers, dtype=int)
        self._votes_stego = np.zeros(num_stegos, dtype=int)

        # Random number generator for breaking ties
        self._rng_for_ties = np.random.RandomState(1)

        # History of OOB errors, one per base learner
        self._errors: typing.List[float] = []

    @property
    def errors(self) -> typing.List[float]:
        """History of OOB errors, one per base learner added so far."""
        return self._errors

    @property
    def error(self) -> typing.Optional[float]:
        """Current OOB error, or ``None`` before the first :meth:`update`."""
        return self._errors[-1] if self._errors else None

    def __len__(self) -> int:
        """Number of base learners added so far."""
        return len(self._errors)

    def update(
        self,
        oob_indices: np.ndarray,
        scores_cover: np.ndarray,
        scores_stego: np.ndarray,
    ) -> float:
        """Add the votes of one base learner and compute the new OOB error.

        :param oob_indices: indices of the out-of-bag rows of this base learner,
            i.e., those not drawn into its bootstrap sample. Rows of covers and
            stegos share the same indices (paired data).
        :param scores_cover: scores of the base learner on the covers at ``oob_indices``.
            Only the sign is used: positive votes stego, negative votes cover.
        :param scores_stego: scores of the base learner on the stegos at ``oob_indices``.
        :return: the new OOB error, which is also appended to :attr:`errors`.
        """
        # Add the votes of this base learner
        self._votes_cover[oob_indices] += np.sign(scores_cover).astype(int)
        self._votes_stego[oob_indices] += np.sign(scores_stego).astype(int)

        # Covers with a positive vote sum are false alarms (classified as stego).
        # Stegos with a negative vote sum are missed detections (classified as cover).
        # Undecided samples (vote sum 0) are resolved at random. Draw first for
        # the covers, then for the stegos; the order matters for reproducibility.
        num_false_alarms = np.sum(self._votes_cover > 0)
        num_undecided = np.sum(self._votes_cover == 0)
        num_false_alarms += np.sum(self._rng_for_ties.rand(num_undecided) > 0.5)

        num_missed_detections = np.sum(self._votes_stego < 0)
        num_undecided = np.sum(self._votes_stego == 0)
        num_missed_detections += np.sum(self._rng_for_ties.rand(num_undecided) < 0.5)

        # See Eq. 5.
        error = (num_false_alarms + num_missed_detections) / (self.num_covers + self.num_stegos)
        self._errors.append(error)
        return error
