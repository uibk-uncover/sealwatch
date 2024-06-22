Usage
=====

.. contents:: Table of Contents
   :local:
   :depth: 1

Installation and setup
----------------------

To use ``sealwatch``, first install it using pip:

.. code-block:: console

   $ pip3 install sealwatch

Import the package with

>>> import sealwatch as sw

For steganalysis on pixels, load the image using `pillow` and `numpy`.

>>> import numpy as np
>>> from PIL import Image
>>> spatial_cover = np.array(Image.open("cover.png"))

For JPEG steganalysis, load the DCT coefficients using our sister project `jpeglib`.

>>> import jpeglib
>>> im_dct = jpeglib.read_dct("cover.jpeg")
>>> im_spatial = jpeglib.read_spatial(  # for J-UNIWARD
...   "cover.jpeg")


.. note::

   ``sealwatch`` expects the DCT coefficients in 4D shape [num_vertical_blocks, num_horizontal_blocks, 8, 8].
   If you use to 2D DCT representation (as used by jpegio, for instance),
   you have to convert it to 4D and back as follows.

   >>> dct_coeffs_4d = (dct_coeffs_2d  # 4D to 2D
   ...   .reshape(dct_coeffs_2d.shape[0]//8, 8, dct_coeffs_2d.shape[1]//8, 8)
   ...   .transpose(0, 2, 1, 3))

   >>> dct_coeffs_2d = (dct_coeffs_4d  # 4D to 2D
   ...   .transpose(0, 2, 1, 3)
   ...   .reshape(dct_coeffs_4d.shape[0]*8, dct_coeffs_4d.shape[1]*8))


*to be completed*