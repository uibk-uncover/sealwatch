"""Ensemble classifier for steganalysis (Kodovsky et al., 2012).

Public interface:

- :class:`FldEnsembleClassifier`, a scikit-learn-style ensemble of Fisher linear
  discriminant (FLD) base learners on random subspaces.
- :class:`Implementation` and the constants :data:`FLD_LEGACY` and
  :data:`FLD_OPTIMIZED`, which select the code path. Both produce the same output.
- :class:`FldEnsembleTrainer`, the deprecated trainer interface of earlier releases.

The original implementation is kept as :mod:`sealwatch.ensemble_classifier_legacy`.

Authors: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

from ._implementation import Implementation
from ._classifier import FldEnsembleClassifier
from ._trainer import FldEnsembleTrainer

FLD_LEGACY = Implementation.FLD_LEGACY
FLD_OPTIMIZED = Implementation.FLD_OPTIMIZED
