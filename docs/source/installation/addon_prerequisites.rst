Before Building an Add-On
=========================

Generalized ADCS has two optional C++ add-ons for trajectory optimization:

- the :doc:`trajectory planner <trajectory_planner_addon>`
  (``tplaunch`` / ``pysat``), from the
  `patrickmckeen/OldPlanner <https://github.com/patrickmckeen/OldPlanner>`__
  repository, and
- :doc:`SALTRO <saltro_addon>` (``saltro_py``), from the
  `nscheuer/SALTRO <https://github.com/nscheuer/SALTRO>`__ repository.

Neither is needed for the pure-Python framework. Both are compiled from source,
so they need a clone of the repository rather than a pip install. See
:doc:`release note 0.1.0 <../release_notes/0_1_0_saltro>` for an overview of
what trajectory optimization adds and where to start.

1. Clone with submodules
------------------------

Both add-ons are git submodules of Generalized ADCS, at ``OldPlanner/`` and
``SALTRO/``. Clone with ``--recurse-submodules``, or initialise them in an
existing clone:

.. code-block:: bash

   git submodule update --init --recursive OldPlanner SALTRO

Re-run this after pulling new commits if the submodule pointer changed.

2. Set up a Python environment with the build tools
---------------------------------------------------

Create and activate a virtual environment in the repository root as on the
:doc:`Installation <index>` page, then install the package with the C++ build
toolchain (CMake, Cython and pybind11). Either use the ``cpp`` extra:

.. code-block:: bash

   pip install -e ".[cpp]"

or install the pinned development environment, which also carries the add-on
build tooling:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

.. note::

   Platform specifics:

   - **Windows:** the add-on builds need **Python 3.13+** on ``PATH``, with
     Visual Studio's C++ workload. Check the version PowerShell uses with
     ``python --version``; if it does not show 3.13 but you are confident it is
     installed, run it explicitly with ``py -3.13 --version``.
   - **WSL:** see :doc:`wsl` for setting up WSL and VS Code, and keep the
     repository on the Linux filesystem.
   - **macOS:** add-on builds on macOS are **community-supported**; see the
     macOS pages for each add-on.

3. Build the add-on
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Platform
     - Trajectory planner
     - SALTRO
   * - Linux / WSL
     - :doc:`trajectory_planner_linux`
     - :doc:`saltro_linux`
   * - Windows
     - :doc:`trajectory_planner_windows`
     - :doc:`saltro_windows`
   * - macOS (community-supported)
     - :doc:`trajectory_planner_macos`
     - :doc:`saltro_macos`

Where Generalized ADCS looks for the build
------------------------------------------

The add-ons are imported through
``ADCS.controller.helpers.optional_dependencies``, which loads the compiled
modules from fixed directories in the repository:

- the trajectory planner's ``tplaunch`` and ``pysat`` modules from
  ``OldPlanner/build/``
  (``get_trajectory_planner_modules()``);
- ``saltro_py`` from ``SALTRO/build/`` (``get_saltro_module()``).

If a module is missing, the error message names the directory it expected.
The compiled module must also match the Python version of the environment you
run Generalized ADCS in.
