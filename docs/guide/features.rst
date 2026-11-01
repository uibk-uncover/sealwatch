Handcrafted features
====================

Handcrafted features describe an image by statistics of its noise residuals,
in which the embedding changes are easier to spot.
Each feature set has a function ``extract``, taking an array,
and ``extract_from_file``, taking a path,
see :doc:`../reference/features`.


Spatial features
----------------

Spatial features, such as SPAM and SRM, are extracted from pixels.

.. code-block:: python

   f = sw.spam.extract(x0)


JPEG features
-------------

JRM and cc-JRM are extracted from the quantized DCT coefficients.

.. code-block:: python

   f = sw.ccjrm.extract(jpeg0.Y, qt=jpeg0.qt[0])

DCTR, PHARM and GFR are extracted from the decompressed pixels, without the final rounding,
with a quantization depending on the JPEG quality factor.
Extracting from the file handles both.

.. code-block:: python

   f = sw.dctr.extract_from_file("cover.jpeg", qf=75)


Submodels
---------

Most feature sets consist of submodels, returned as a dict.
For a classifier, flatten them to a vector.

.. code-block:: python

   f = sw.spam.extract(x0)
   f = sw.tools.flatten(f)
