Detectors
=========

Ensemble of FLD
---------------

.. autoclass:: sealwatch.ensemble_classifier.EnsembleClassifier
   :members: num_base_learners, d_sub, predict_confidence, predict, score

.. autoclass:: sealwatch.ensemble_classifier.FldEnsembleTrainer
   :members: train


Helper functions
----------------

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_hdf5

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_features

.. autofunction:: sealwatch.ensemble_classifier.helpers.remove_file_extension

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_and_split_features

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_features_subset
