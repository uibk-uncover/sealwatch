Handcrafted features
====================

HCF-COM
-------

.. autofunction:: sealwatch.hcfcom.extract

.. autofunction:: sealwatch.hcfcom.extract_from_file


SPAM
----

.. autofunction:: sealwatch.spam.extract

.. autofunction:: sealwatch.spam.extract_from_file


SRM
---

.. autofunction:: sealwatch.srm.extract

.. autofunction:: sealwatch.srm.extract_from_file

.. autofunction:: sealwatch.srmq1.extract

.. autofunction:: sealwatch.srmq1.extract_from_file


CRM
---

.. autofunction:: sealwatch.crm.extract

.. autofunction:: sealwatch.crm.extract_from_file


JRM
---

.. autofunction:: sealwatch.jrm.extract

.. autofunction:: sealwatch.jrm.extract_from_file

.. autofunction:: sealwatch.ccjrm.extract

.. autofunction:: sealwatch.ccjrm.extract_from_file


DCTR
----

.. autofunction:: sealwatch.dctr.extract

.. autofunction:: sealwatch.dctr.extract_from_file


PHARM
-----

.. autofunction:: sealwatch.pharm.extract

.. autofunction:: sealwatch.pharm.extract_from_file

.. autoclass:: sealwatch.pharm.Implementation
   :members: PHARM_ORIGINAL, PHARM_FIX


GFR
---

.. autofunction:: sealwatch.gfr.extract

.. autofunction:: sealwatch.gfr.extract_from_file

.. autofunction:: sealwatch.gfr.compute_gabor_kernels

.. autoclass:: sealwatch.gfr.Implementation
   :members: GFR_ORIGINAL, GFR_FIX


.. _feature-files:

Feature files
-------------

Features of cover and stego images, loaded in pairs, from HDF5 files or zarr stores.

.. autofunction:: sealwatch.tools.load_features

.. autofunction:: sealwatch.tools.load_and_split_features

.. autofunction:: sealwatch.tools.load_features_subset

.. autofunction:: sealwatch.tools.load_hdf5

.. autofunction:: sealwatch.tools.remove_file_extension


Feature store
-------------

Features of many images in a zarr store, written in parallel by the tasks of a job array (e.g., SLURM),
one file per task. The loaders of feature files read it as an HDF5 file.

.. automodule:: sealwatch.tools._zarr
   :no-members:

.. autofunction:: sealwatch.tools.create_feature_store

.. autofunction:: sealwatch.tools.num_feature_tasks

.. autofunction:: sealwatch.tools.feature_task_rows

.. autofunction:: sealwatch.tools.write_feature_task

.. autofunction:: sealwatch.tools.missing_feature_tasks

.. autofunction:: sealwatch.tools.load_zarr
