SALTRO macOS
============

``saltro_py`` is an optional add-on for Generalized ADCS.

.. note::

   Add-on builds on macOS are **community-supported**. The steps below are
   verified on Apple Silicon (Homebrew prefix ``/opt/homebrew``). On Intel Macs,
   substitute the Homebrew prefix ``/usr/local`` wherever ``/opt/homebrew``
   appears.

The official `SALTRO installation page
<https://nscheuer.github.io/SALTRO/datasheets/Installation.html>`__ does not yet
cover macOS.

Prerequisites
-------------
Set up the toolchain and a Python 3.10+ virtual environment in the
``Generalized_ADCS/`` folder as described under *Prerequisites* and *Create
the virtual environment* on the :doc:`trajectory_planner_macos` page (Xcode
Command Line Tools, Homebrew, pyenv, ``pip install -r requirements.txt`` and
``pip install -e . --no-deps --no-build-isolation``).

Initialise the submodule if needed:

.. code-block:: bash

   git submodule update --init --recursive SALTRO

Build
-----
Run with the virtual environment active, from the repository root:

.. code-block:: bash

   cd SALTRO && mkdir -p build && cd build
   cmake .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DSALTRO_WARNINGS_AS_ERRORS=OFF \
     -DPYBIND11_FINDPYTHON=ON \
     -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
     -DPython_ROOT_DIR="$VIRTUAL_ENV" \
     -DPython_FIND_VIRTUALENV=ONLY
   cmake --build . -j$(sysctl -n hw.logicalcpu)

Two macOS-specific notes:

- ``-DSALTRO_WARNINGS_AS_ERRORS=OFF`` is required: SALTRO's default
  ``-Werror -Wconversion -Wshadow`` does not pass under Apple ``clang``.
- The ``Python_ROOT_DIR`` / ``Python_FIND_VIRTUALENV=ONLY`` hints are required
  because the bundled pybind11 (v2.12.0) otherwise silently selects the Xcode
  framework Python 3.9 and produces a ``cpython-39`` module the virtual
  environment cannot import. Verify the artifact is
  ``SALTRO/build/saltro_py.cpython-312-darwin.so`` (for a Python 3.12
  environment).

Debugging
---------
macOS uses LLDB (not GDB), for example:

.. code-block:: bash

   lldb -- python examples/tutorials/advanced_topics/saltro_godot.py
