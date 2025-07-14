Reference
=========

*to be completed*

For more information, see the `glossary <https://sealwatch.readthedocs.io/en/latest/glossary.html#steganographic-design>`__.

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

.. autofunction:: sealwatch.ws.unet_estimator


Histogram attack
""""""""""""""""

.. autofunction:: sealwatch.F5.attack


RJCA
""""

.. autofunction:: sealwatch.rjca.attack


Handcrafted features
--------------------


HCF-COM
"""""""

.. autofunction:: sealwatch.hcfcom.extract

.. autofunction:: sealwatch.hcfcom.extract_from_file


SPAM
""""

.. autofunction:: sealwatch.spam.extract

.. autofunction:: sealwatch.spam.extract_from_file

.. autofunction:: sealwatch.spam_rs.extract

SRM
""""

.. autofunction:: sealwatch.srm.extract

.. autofunction:: sealwatch.srm.extract_from_file

.. autofunction:: sealwatch.srmq1.extract

.. autofunction:: sealwatch.srmq1.extract_from_file


CRM
""""

.. autofunction:: sealwatch.crm.extract

.. autofunction:: sealwatch.crm.extract_from_file


JRM
""""

.. autofunction:: sealwatch.jrm.extract

.. autofunction:: sealwatch.jrm.extract_from_file

.. autofunction:: sealwatch.ccjrm.extract

.. autofunction:: sealwatch.ccjrm.extract_from_file


DCTR
""""

.. autofunction:: sealwatch.dctr.extract

.. autofunction:: sealwatch.dctr.extract_from_file


PHARM
"""""

.. autofunction:: sealwatch.pharm.extract

.. autofunction:: sealwatch.pharm.extract_from_file

.. autoclass:: sealwatch.pharm.Implementation
   :members: PHARM_ORIGINAL, PHARM_REVISITED

GFR
"""

.. autofunction:: sealwatch.gfr.extract

.. autofunction:: sealwatch.gfr.extract_from_file

.. autoclass:: sealwatch.gfr.Implementation
   :members: GFR_ORIGINAL, GFR_FIX

Detectors
---------

.. autoclass:: sealwatch.ensemble_classifier.EnsembleClassifier
   :members: num_base_learners, d_sub, predict_confidence, predict, score

.. autoclass:: sealwatch.ensemble_classifier.FldEnsembleTrainer
   :members: train


.. autoclass:: sealwatch.xunet.XuNet
   :members: forward

.. autofunction:: sealwatch.xunet.pretrained

.. autofunction:: sealwatch.xunet.infere_single

Helper functions
""""""""""""""""

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_hdf5

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_features

.. autofunction:: sealwatch.ensemble_classifier.helpers.remove_file_extension

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_and_split_features

.. autofunction:: sealwatch.ensemble_classifier.helpers.load_features_subset
