SALTRO Linux / WSL
==================

``saltro_py`` is an optional add-on for Generalized_ADCS.

The commands below are adapted from `SALTRO <https://nscheuer.github.io/SALTRO>`_'s
in-repository installation guide:

- `SALTRO/docs/INSTALL.md <../../../SALTRO/docs/INSTALL.md>`_

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
