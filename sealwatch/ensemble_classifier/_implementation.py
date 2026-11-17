"""Selection of the ensemble classifier implementation."""

import enum


class Implementation(enum.Enum):
    """Ensemble classifier implementation to choose from.

    ``FLD_LEGACY`` matches the legacy implementation exactly for the same seed.
    Whether the search for the subspace dimensionality is exhaustive is controlled
    by the ``search_L`` parameter of :class:`FldEnsembleClassifier`, not by this enum.
    """

    FLD_LEGACY = enum.auto()
    """Exact match: LU solve and Matlab-compatible random subspaces. Optimized where the result stays identical."""
    # An FLD_OPTIMIZED (Cholesky solve, rng.choice subspaces) was tried and dropped: its models
    # differ from the legacy ones, and the gain was small at typical sizes. See the comments in
    # FisherLinearDiscriminant._solve and FldEnsembleClassifier._generate_random_subspace.
