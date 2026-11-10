"""Selection of the ensemble classifier implementation."""

import enum


class Implementation(enum.Enum):
    """Ensemble classifier implementation to choose from.

    All members produce the same output for the same input and seed (and match
    the reference Matlab implementation). They differ only in how the result is
    computed. Whether the result is exact or approximate is controlled by the
    ``search_L`` parameter of :class:`FldEnsembleClassifier`, not by this enum.
    """

    FLD_LEGACY = enum.auto()
    """Original implementation, kept as a reference (:mod:`sealwatch.ensemble_classifier_legacy`)."""
    FLD_OPTIMIZED = enum.auto()
    """Optimized implementation: Cholesky solve, vectorized threshold search, no redundant copies."""
