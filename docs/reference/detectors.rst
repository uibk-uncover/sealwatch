Detectors
=========

Ensemble of FLD
---------------

.. autoclass:: sealwatch.ensemble_classifier.FldEnsembleClassifier
   :members: fit, decision_function, predict, predict_confidence, num_base_learners

.. autoclass:: sealwatch.ensemble_classifier.FisherLinearDiscriminant
   :members: fit, fit_presplit, predict, score

.. autoclass:: sealwatch.ensemble_classifier.Implementation
   :members: FLD_LEGACY


Deprecated
----------

The interface of releases up to 2026.11, kept for backward compatibility.
It warns when used, and gives the same results as before.
The loaders of feature files in ``sealwatch.ensemble_classifier.helpers``
moved to :mod:`sealwatch.tools`, see :ref:`feature-files`; the old names warn and call them.

.. autoclass:: sealwatch.ensemble_classifier.FldEnsembleTrainer
   :members: train

.. autoclass:: sealwatch.ensemble_classifier.EnsembleClassifier
   :members: num_base_learners, d_sub, predict_confidence, predict, score
