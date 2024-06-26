sealwatch
=========

**sealwatch** is a Python package, containing implementations of modern image steganalysis algorithms.

.. warning::

   This project is under intensive development as we speak.

Catching steganography has never been easier!

>>> # load suspicious image
>>> import numpy as np
>>> from PIL import Image
>>> img = np.array(Image.open("suspicious.png"))
>>>
>>> # apply WS steganalysis
>>> import sealwatch as sw
>>> estimated_change_rate = sw.ws.attack(img)
>>> print(estimated_change_rate > 0.)


.. list-table:: Available steganalysis algorithms.
   :widths: 25 75
   :width: 100%
   :header-rows: 1

   * - Type
     - Algorithms
   * - Analytical attacks
     - chi2, SPA, WS, RJCA
   * - Features
     - SPAM, JRM, DCTR, PHARM, GFR
   * - Detectors
     - FLD

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   reference
   glossary
   faq