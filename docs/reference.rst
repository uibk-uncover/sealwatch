Reference
=========

*to be completed*

For more information, see the `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#steganographic-design>`__.

.. contents:: Table of Contents
   :local:
   :depth: 2

Analytical attacks
------------------

$\chi^2$-attack
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

.. autofunction:: sealwatch.features.pharm_original.extract_pharm_original_features_from_file

.. autofunction:: sealwatch.features.pharm_original.extract_pharm_original_features_from_img

.. autofunction:: sealwatch.features.pharm_revisited.extract_pharm_revisited_features_from_file

.. autofunction:: sealwatch.features.pharm_revisited.extract_pharm_revisited_features_from_img




GFR
""""

.. autofunction:: sealwatch.features.gfr.extract_gfr_features_from_file

.. autofunction:: sealwatch.features.gfr.extract_gfr_features_from_img



Detectors
---------

