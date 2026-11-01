Analytical attacks
==================

Analytical attacks estimate the payload of a specific embedding, without training.
They are targeted, i.e., they detect only the embedding they are designed for.


Detecting LSB replacement
-------------------------

The WS attack estimates the change rate of LSB replacement,
here simulated with `conseal <https://conseal.readthedocs.io/>`__.

.. code-block:: python

   import conseal as cl
   import numpy as np
   from PIL import Image
   import sealwatch as sw

   x0 = np.array(Image.open("cover.png").convert("L"))
   beta_hat = sw.ws.attack(x0)  # about 0

   # LSB replacement at 0.4 bits per pixel, changes about 20% of the pixels
   x1 = cl.lsb.simulate(x0, .4, modify=cl.LSB_REPLACEMENT, seed=12345)
   beta_hat = sw.ws.attack(x1)  # about 0.2

The estimate is the change rate, i.e., half of the embedding rate,
as LSB replacement changes half of the pixels it embeds into.


Targeted attacks
----------------

LSB matching changes the pixels by +1 or -1, instead of flipping the LSB.
It changes as many pixels as LSB replacement, but WS does not detect it.

.. code-block:: python

   # LSB matching at 0.4 bits per pixel, changes about 20% of the pixels
   x1 = cl.lsb.simulate(x0, .4, modify=cl.LSB_MATCHING, seed=12345)
   beta_hat = sw.ws.attack(x1)  # about 0, as for the cover

The chi2 attack and SPA also target LSB replacement,
and the histogram attack targets F5.
To detect other embedding, train a detector, see :doc:`detector`.
