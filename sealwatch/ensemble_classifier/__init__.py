"""Ensemble classifier for steganalysis (Kodovsky et al., 2012).

Public interface:

- :class:`FldEnsembleClassifier`, a scikit-learn-style ensemble of Fisher linear
  discriminant (FLD) base learners on random subspaces.
- :class:`Implementation` and the constant :data:`FLD_LEGACY`, which selects the code path.
  ``FLD_LEGACY`` matches the legacy implementation exactly.
- :class:`FisherLinearDiscriminant`, the base learner, usable on its own.
- :class:`FldEnsembleTrainer` and :class:`EnsembleClassifier`, deprecated wrappers that keep
  the interface of earlier releases (separate covers and stegos, labels -1 and +1).
- :mod:`~sealwatch.ensemble_classifier.helpers`, deprecated loaders of feature files,
  moved to :mod:`sealwatch.tools`.

Authors: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

from ._implementation import Implementation
from ._classifier import FldEnsembleClassifier
from .fld import FisherLinearDiscriminant
from ._deprecated import EnsembleClassifier, FldEnsembleTrainer
from . import helpers

FLD_LEGACY = Implementation.FLD_LEGACY
