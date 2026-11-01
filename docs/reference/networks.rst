Neural networks
===============

The neural networks are in ``sealwatch.nn``, imported on first use, as they require torch.

XuNet
-----

.. autoclass:: sealwatch.nn.XuNet
   :members: forward, pretrained


SRNet
-----

.. autoclass:: sealwatch.nn.SRNet
   :members: forward, pretrained


ZhuNet
------

.. autoclass:: sealwatch.nn.ZhuNet
   :members: forward


UcNet
-----

.. autoclass:: sealwatch.nn.UcNet
   :members: forward, pretrained


EfficientNet
------------

.. autoclass:: sealwatch.nn.B0
   :members: forward, pretrained

.. autoclass:: sealwatch.nn.B4
   :members: forward, pretrained


PENet
-----

.. autoclass:: sealwatch.nn.PENet
   :members: forward, pretrained

.. autoclass:: sealwatch.nn.penet.Implementation
   :members: PENET_ORIGINAL, PENET_FIX


Inference
---------

.. autofunction:: sealwatch.nn.infere_single
