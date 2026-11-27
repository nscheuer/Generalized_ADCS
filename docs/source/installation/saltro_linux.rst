SALTRO Linux / WSL
==================

``saltro_py`` is an optional add-on for Generalized_ADCS.

The commands below are adapted from `SALTRO <https://nscheuer.github.io/SALTRO>`_'s
in-repository installation guide:

- `SALTRO/docs/INSTALL.md <https://github.com/nscheuer/SALTRO/blob/main/docs/INSTALL.md>`__

Complete :doc:`addon_prerequisites` first; on Windows, see :doc:`wsl`.

Initialise the submodule first if needed:

.. code-block:: bash

   git submodule update --init --recursive SALTRO

Prerequisites
-------------
Install required system packages:

.. code-block:: bash

   sudo apt update
   sudo apt upgrade
   sudo apt install cmake ccache
   sudo apt install -y build-essential python3-dev
   sudo apt install python3-venv

Virtual Environment
-------------------
From the ``SALTRO/`` directory:

.. code-block:: bash

   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

Compile (Release)
-----------------

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSALTRO_BUILD_PYTHON=ON
   cmake --build build -j

Compile (Debug)
---------------

.. code-block:: bash

   sudo apt install gdb
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSALTRO_BUILD_PYTHON=ON
   cmake --build build -j

Test
----

.. code-block:: bash

   cd build
   ctest --output-on-failure
   cd ../tests
   pytest -q

Generalized ADCS imports ``saltro_py`` from ``SALTRO/build/``, the directory
the commands above build into. Build it for the same Python version as the
environment you run Generalized ADCS in.
