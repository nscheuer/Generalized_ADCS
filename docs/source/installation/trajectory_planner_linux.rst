Trajectory Planner (Linux / WSL)
================================

The ``trajectory_planner`` module is a C++ library built with Pybind and imported
into the Python codebase. It must be compiled from source on Linux and WSL.

Canonical instructions (including both Windows and Linux) are maintained in:

- `docs/Install_Trajectory_Planner.md <../../Install_Trajectory_Planner.md>`_

For algorithm background, see:

- `https://nscheuer.github.io/SALTRO <https://nscheuer.github.io/SALTRO>`_

Prerequisites
-------------
Ensure the following system packages are installed:

.. code-block:: bash

   sudo apt update
   sudo apt install -y \
     cmake \
     g++ \
     libarmadillo-dev \
     libboost-math-dev

Build Instructions
------------------
From the repository root:

.. code-block:: bash

   cd trajectory_planner
   mkdir -p build
   cd build

Configure the build:

.. code-block:: bash

   cmake .. \
     -DCMAKE_BUILD_TYPE=Debug \
     -DPython3_EXECUTABLE=$(which python3) \
     -DPYTHON_EXECUTABLE=$(which python3) \
     -DPYTHON_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
     -DPYTHON_INCLUDE_DIRS=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
     -DPYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.12.so \
     -DPYTHON_LIBRARIES=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.12.so

Compile:

.. code-block:: bash

   make -j$(nproc)

Debugging
---------
To debug the C++ extension, install GDB:

.. code-block:: bash

   sudo apt install gdb

Verify installation:

.. code-block:: bash

   which gdb
