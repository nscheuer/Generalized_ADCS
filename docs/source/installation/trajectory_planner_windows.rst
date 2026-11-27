Trajectory Planner Windows
==========================

``trajectory_planner`` is an optional add-on for Generalized ADCS.

The module is a C++ project built with Pybind and imported into the Python
codebase. On Windows, it must be compiled using **Windows CMake** and
**vcpkg**. Complete :doc:`addon_prerequisites` first.

For algorithm background, see the
`SALTRO documentation website <https://nscheuer.github.io/SALTRO>`__.

Prerequisites
-------------
- `Visual Studio <https://learn.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2022>`__
  with the C++ workload
- **Python 3.13+**, with the Python executable on the system
  `PATH <https://realpython.com/add-python-to-path/>`__
- Python virtual environment activated
- Windows CMake (not MSYS)
- vcpkg

Ensure the correct Python version
---------------------------------
Check the version of Python that PowerShell uses:

.. code-block:: powershell

   python --version

If PowerShell does not display your installed Python 3.13+, but you are
confident it is installed on your machine, run it explicitly:

.. code-block:: powershell

   py -3.13 --version

Ensure Correct CMake
--------------------
Verify which CMake executable is being used:

.. code-block:: powershell

   Get-Command cmake | Select-Object Source

If the output points to MSYS (e.g. ``C:\msys64\mingw64\bin\cmake.exe``),
install Windows CMake:

.. code-block:: powershell

   winget install Kitware.CMake

The Windows CMake executable is typically installed at:

``C:\Program Files\CMake\bin\cmake.exe``

Install vcpkg
-------------
If ``vcpkg`` is not installed:

.. code-block:: powershell

   cd $env:USERPROFILE
   git clone https://github.com/microsoft/vcpkg
   cd vcpkg
   .\bootstrap-vcpkg.bat

Install Dependencies
--------------------
Install the required C++ libraries. The full path works whether or not
``vcpkg`` is on your ``PATH``:

.. code-block:: powershell

   & "$env:USERPROFILE\vcpkg\vcpkg" install armadillo
   & "$env:USERPROFILE\vcpkg\vcpkg" install boost-math:x64-windows

Note that this step may take up to 15 minutes.

Build Instructions
------------------
The trajectory_planner C++ source lives in the
`OldPlanner <https://github.com/patrickmckeen/OldPlanner>`__ submodule.
Initialise it first if needed:

.. code-block:: powershell

   git submodule update --init --recursive OldPlanner

Then create the ``OldPlanner`` build directory and set the tool paths.
``$PYTHON_VENV`` must point at your virtual environment's ``python.exe``; the
path below matches the ``.venv`` created on the :doc:`Installation <index>`
page (use ``venv`` instead if that is what you named it):

.. code-block:: powershell

   cd OldPlanner
   mkdir build
   cd build

   $CMAKE_EXE = "C:\Program Files\CMake\bin\cmake.exe"
   $VCPKG_ROOT = "$env:USERPROFILE\vcpkg"
   $PYTHON_VENV = "$PWD\..\..\.venv\Scripts\python.exe"

Configure the build:

.. code-block:: powershell

   & $CMAKE_EXE .. `
     -A x64 `
     -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
     -DVCPKG_TARGET_TRIPLET=x64-windows `
     -DPython3_EXECUTABLE="$PYTHON_VENV"

Build the project:

.. code-block:: powershell

   & $CMAKE_EXE --build . --config Release

Generalized ADCS loads the compiled ``tplaunch`` and ``pysat`` modules from
``OldPlanner/build/``.
