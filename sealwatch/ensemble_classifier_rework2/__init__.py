"""Performance-oriented FLD ensemble for binary steganalysis.

This sub-package provides :class:`FldEnsembleClassifier`, a scikit-learn-compatible
re-implementation of the ensemble classifier from:

    J. Kodovsky, J. Fridrich, V. Holub, "Ensemble Classifiers for Steganalysis
    of Digital Media", IEEE TIFS 7(2), pp. 432-444, 2012.

The key difference to :mod:`sealwatch.ensemble_classifier` is a *capped-L subspace
search* (controlled by the ``search_effort`` parameter) that avoids fully converging
every discarded ``d_sub`` candidate, yielding a 2-5x training speedup while
preserving the detection error within run-to-run noise of the original.

See :class:`FldEnsembleClassifier` for full documentation and usage examples.
"""

from .fld_ensemble_trainer import (  # noqa: F401
    FldEnsembleClassifier,
    FldEnsembleTrainer,
)
