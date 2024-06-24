Reference
=========

*to be completed*

For more information, see the `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#steganographic-design>`__.

.. contents:: Table of Contents
   :local:
   :depth: 2

Analytical attacks
------------------

chi2 attack
"""""""""""""""

.. autofunction:: sealwatch.chi2.attack


SPA
"""

.. autofunction:: sealwatch.spa.attack


WS
""""

.. autofunction:: sealwatch.ws.attack


RJCA
""""

.. autofunction:: sealwatch.rjca.attack

Handcrafted features
--------------------

SPAM
""""

.. autofunction:: sealwatch.features.spam.extract_spam686_features_from_img

.. autofunction:: sealwatch.features.spam.extract_spam686_features_from_file


JRM
""""

.. autofunction:: sealwatch.features.jrm.extract_jrm_features_from_img

.. autofunction:: sealwatch.features.jrm.extract_jrm_features_from_file



DCTR
""""

.. autofunction:: sealwatch.features.dctr.extract_dctr_features_from_img

.. autofunction:: sealwatch.features.dctr.extract_dctr_features_from_file


PHARM
""""""

.. autofunction:: sealwatch.features.pharm.extract_pharm_original_features_from_file

.. autofunction:: sealwatch.features.pharm.extract_pharm_original_features_from_img

.. autofunction:: sealwatch.features.pharm.extract_pharm_revisited_features_from_file

.. autofunction:: sealwatch.features.pharm.extract_pharm_revisited_features_from_img




GFR
""""

.. autofunction:: sealwatch.features.gfr.extract_gfr_features_from_file

.. autofunction:: sealwatch.features.gfr.extract_gfr_features_from_img


Detectors
---------

.. autoclass:: conseal.ensemble_classifier.EnsembleClassifier
   :members: num_base_learners, d_sub, predict_confidence, predict, score

.. autoclass:: conseal.ensemble_classifier.FldEnsembleTrainer
   :members: train

Helper functions
""""""""""""""""

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_hdf5

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_features

.. autofunction:: sealwatch.ensemble_classifier.helpers.remove_file_extension

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_and_split_features

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_features_subset
