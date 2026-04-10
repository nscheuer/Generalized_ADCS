Trajectory Planner (Windows)
============================

The ``trajectory_planner`` module is a C++ project built with Pybind and imported
into the Python codebase. On Windows, it must be compiled using **Windows CMake**
and **vcpkg**.

Canonical instructions (including both Windows and Linux) are maintained in:

- `docs/Install_Trajectory_Planner.md <../../Install_Trajectory_Planner.md>`_

For algorithm background, see:

- `https://nscheuer.github.io/SALTRO <https://nscheuer.github.io/SALTRO>`_

Prerequisites
-------------
- Visual Studio (C++ workload)
- Python virtual environment activated
- Windows CMake (not MSYS)

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
Install required C++ libraries:

.. code-block:: powershell

   vcpkg install armadillo
   vcpkg install boost-math:x64-windows

Note that this step may take up to 15 minutes.

Build Instructions
------------------
From the ``trajectory_planner`` build directory:

.. code-block:: powershell

   $CMAKE_EXE = "C:\Program Files\CMake\bin\cmake.exe"
   $VCPKG_ROOT = "$env:USERPROFILE\vcpkg"
   $PYTHON_VENV = ".\venv\Scripts\python.exe"

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
