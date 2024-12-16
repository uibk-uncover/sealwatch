sealwatch
=========

**sealwatch** is a Python package, containing implementations of modern image steganalysis algorithms.

.. note::

   This project is under active development.

Catching steganography has never been easier!

>>> # load suspicious image
>>> import numpy as np
>>> from PIL import Image
>>> img = np.array(Image.open("suspicious.png"))
>>>
>>> # apply WS steganalysis
>>> import sealwatch as sw
>>> beta_hat = sw.ws.attack(img)  # estimate change rate
>>> print(beta_hat > 0.)

>>> # load suspicious image
>>> import jpeglib
>>> jpeg1 = jpeglib.read_dct("suspicious.jpeg")
>>>
>>> # extract JRM features
>>> features = sw.ccjrm.extract(y1=jpeg1.Y, qt=jpeg1.qt[0])
>>> model = pickle.load(open('model.pickle', 'rb'))  # load trained FLD
>>> y_hat = model.predict(features[None])  # predict cover/stego label
>>> assert y_hat != 1

.. list-table:: Available steganalysis algorithms.
   :widths: 25 75
   :width: 100%
   :header-rows: 1

   * - Type
     - Algorithms
   * - Analytical attacks
     - chi2, F5, SPA, WS, RJCA
   * - Features
     - CRM, cc-JRM, HCF-COM, JRM, DCTR, PHARM, GFR, SPAM, SRM
   * - Detectors
     - ensemble of FLD

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   reference
   glossary
   faq