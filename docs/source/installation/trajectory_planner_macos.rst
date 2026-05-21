Trajectory Planner macOS
=========================

``trajectory_planner`` is an optional add-on for Generalized_ADCS. The module
is a C++ project built with pybind11 and imported into the Python codebase.

Canonical instructions (including Windows and Linux) are maintained in:

- `docs/Install_Trajectory_Planner.md <../../Install_Trajectory_Planner.md>`_

For algorithm background see:

- `SALTRO documentation website <https://nscheuer.github.io/SALTRO>`_

Prerequisites
-------------
- Homebrew
- Python virtual environment activated

Install system dependencies::

   brew install armadillo boost

Note: on Intel Macs replace ``/opt/homebrew`` with ``/usr/local`` below.

Build trajectory_planner
------------------------
The trajectory_planner C++ source lives in the
`OldPlanner <https://github.com/patrickmckeen/OldPlanner>`_ submodule.
Initialise it first if needed:

.. code-block:: bash

   git submodule update --init --recursive OldPlanner

Then with the virtual environment active (so ``$VIRTUAL_ENV`` is set):

.. code-block:: bash

   cd OldPlanner
   mkdir -p build && cd build

   cmake .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
     -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/armadillo;/opt/homebrew/opt/boost;/opt/homebrew" \
     -DCMAKE_CXX_FLAGS="-I/opt/homebrew/include"

   cmake --build . -j$(sysctl -n hw.logicalcpu)

The ``-DCMAKE_CXX_FLAGS=-I/opt/homebrew/include`` flag is required so the
``tp_test`` target can find Boost's ``boost/math`` headers; the CMake project
does not add a Boost include path itself.
