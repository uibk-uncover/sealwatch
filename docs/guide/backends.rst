Backends
========

Some feature extractors have two implementations,
in Python and in Rust.
Rust is the default, and both give bit-identical features,
as tested in the unit tests.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Rust implementation
     - Requirements
   * - SPAM
     - uint8 image
   * - SRM, SRMQ1
     - uint8 image, of at least 5x5 pixels
   * - CRM
     - fixed implementation, uint8 color image, of at least 5x5 pixels
   * - DCTR
     - separable convolutions, image of at least 8x8 pixels
   * - GFR
     - image of at least 8x8 pixels

The other methods are implemented in Python only.
If the input does not meet the requirements,
sealwatch falls back to Python silently.


Switching the backend
---------------------

Switch the backend globally,

.. code-block:: python

   sw.set_backend(sw.BACKEND_PYTHON)

or only for a block of code.

.. code-block:: python

   with sw.BACKEND_PYTHON:
       features = sw.srm.extract(x0)


Installation
------------

The Rust backend comes prebuilt for all major platforms.
``pip install sealwatch`` downloads a Python wheel with the compiled Rust code,
so no Rust is needed to use it.
One wheel per platform serves all supported Python versions.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Platform
     - Architecture
   * - Linux
     - x86-64
   * - macOS
     - Apple Silicon (arm64)
   * - Windows
     - x86-64

On other platforms, pip builds the package from source.
This requires the `Rust toolchain <https://rustup.rs/>`__ to be installed,

.. code-block:: console

   $ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   $ pip install sealwatch

and the same holds for installing from a clone of the repository.

.. code-block:: console

   $ git clone https://github.com/uibk-uncover/sealwatch
   $ cd sealwatch
   $ pip install .
