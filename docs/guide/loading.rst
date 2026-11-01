Loading images
==============

sealwatch works on numpy arrays.
How you get them depends on the domain of the steganalysis.


Installation
------------

Install sealwatch using pip.

.. code-block:: console

   $ pip3 install sealwatch

Import the package with

.. code-block:: python

   import sealwatch as sw


Spatial images
--------------

For steganalysis on pixels, load the image using `pillow <https://pypi.org/project/pillow/>`__ and numpy.

.. code-block:: python

   import numpy as np
   from PIL import Image

   x0 = np.array(Image.open("cover.png"))


JPEG images
-----------

For JPEG steganalysis, load the DCT coefficients using our sister project
`jpeglib <https://jpeglib.readthedocs.io/>`__.

.. code-block:: python

   import jpeglib

   jpeg0 = jpeglib.read_dct("cover.jpeg")
   im0 = jpeglib.read_spatial("cover.jpeg")

.. note::

   sealwatch expects the DCT coefficients in 4D shape [num_vertical_blocks, num_horizontal_blocks, 8, 8].
   If you use the 2D DCT representation (as used by jpegio, for instance),
   you have to convert it to 4D and back as follows.

   .. code-block:: python

      y_4d = sw.tools.jpegio_to_jpeglib(y_2d)
      y_2d = sw.tools.jpeglib_to_jpegio(y_4d)

Alternatively, you can use our custom decompressor, which avoids the final rounding.

.. code-block:: python

   x0 = sw.tools.jpeg.decompress(jpeg0.Y, jpeg0.qt[0])
   x0 = x0[:jpeg0.height, :jpeg0.width]  # crop to image size
