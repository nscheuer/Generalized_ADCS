Trajectory Planner macOS
=========================

``trajectory_planner`` is an optional add-on for Generalized ADCS. The module
is a C++ project built with pybind11 and imported into the Python codebase.

.. note::

   Add-on builds on macOS are **community-supported**. The steps below are
   verified on Apple Silicon (Homebrew prefix ``/opt/homebrew``). On Intel Macs,
   substitute the Homebrew prefix ``/usr/local`` wherever ``/opt/homebrew``
   appears.

For algorithm background see the
`SALTRO documentation website <https://nscheuer.github.io/SALTRO>`__.

Prerequisites
-------------
- macOS (Apple Silicon or Intel)
- `Xcode Command Line Tools <https://mac.install.guide/commandlinetools/>`__
  (provides ``clang`` / ``clang++``): ``xcode-select --install``
- `Homebrew <https://brew.sh>`__
- Python **3.10+**, for example via `pyenv <https://github.com/pyenv/pyenv>`__.
  The macOS *system* Python is typically 3.9.x, which is too old for this
  project (``requires-python >= 3.10``).

Install the toolchain and system dependencies:

.. code-block:: bash

   xcode-select --install                 # if not already installed
   brew install pyenv armadillo boost

``armadillo`` and ``boost`` are required to build ``trajectory_planner``.
``cmake`` is not needed system-wide: it is installed as a pinned pip dependency
into the virtual environment.

Create the virtual environment
------------------------------
Install a supported Python (you do not need to change your global Python; the
virtual environment is created directly from this interpreter):

.. code-block:: bash

   pyenv install 3.12.13

Inside the ``Generalized_ADCS/`` folder:

.. code-block:: bash

   ~/.pyenv/versions/3.12.13/bin/python3 -m venv venv
   source venv/bin/activate
   python --version            # should print 3.12.13

Install the Python dependencies:

.. code-block:: bash

   pip install --upgrade pip wheel setuptools
   pip install -r requirements.txt
   pip install -e . --no-deps --no-build-isolation

The final line registers the ``ADCS`` package itself in editable mode.

Build trajectory_planner
------------------------
The trajectory_planner C++ source lives in the
`OldPlanner <https://github.com/patrickmckeen/OldPlanner>`__ submodule.
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

Debugging
---------
macOS uses LLDB (not GDB):

.. code-block:: bash

   lldb -- python examples/tutorials/advanced_topics/plan_and_track_godot.py
