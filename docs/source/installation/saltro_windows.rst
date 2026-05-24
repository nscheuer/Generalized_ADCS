SALTRO Windows
==============

``saltro_py`` is an optional add-on for Generalized_ADCS.

The steps below mirror the Linux/WSL `SALTRO <https://nscheuer.github.io/SALTRO>`_
flow using native Windows tools.
For the official upstream guide, see:

- `SALTRO installation page <https://nscheuer.github.io/SALTRO/datasheets/Installation.html>`_
- `SALTRO documentation website <https://nscheuer.github.io/SALTRO>`_

Initialise the submodule first if needed:

.. code-block:: powershell

   git submodule update --init --recursive SALTRO

Prerequisites
-------------
- Visual Studio 2022 with C++ workload
- Windows CMake (Kitware)
- Git
- Python 3.13+ on PATH

Verify tools in PowerShell:

.. code-block:: powershell

	cmake --version
	cl
	python --version

If ``cmake`` points to MSYS/MinGW, install and use Windows CMake:

.. code-block:: powershell

	winget install Kitware.CMake

Virtual Environment
-------------------
From the ``SALTRO/`` directory:

.. code-block:: powershell

	python -m venv venv
	.\venv\Scripts\Activate.ps1
	pip install -r requirements.txt

Compile (Release)
-----------------
Configure and build the Python-enabled `SALTRO <https://nscheuer.github.io/SALTRO>`_ module:

.. code-block:: powershell

	$PYTHON_VENV = "$PWD\venv\Scripts\python.exe"
	cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DSALTRO_BUILD_PYTHON=ON -DPython3_EXECUTABLE="$PYTHON_VENV"
	cmake --build build --config Release

Compile (Debug)
---------------
Configure and build debug binaries:

.. code-block:: powershell

	$PYTHON_VENV = "$PWD\venv\Scripts\python.exe"
	cmake -S . -B build-debug -G "Visual Studio 17 2022" -A x64 -DSALTRO_BUILD_PYTHON=ON -DPython3_EXECUTABLE="$PYTHON_VENV"
	cmake --build build-debug --config Debug

Test
----
Run C++ and Python tests:

.. code-block:: powershell

	ctest --test-dir build --output-on-failure -C Release
	cd tests
	pytest -q

Troubleshooting
---------------
- If Python bindings are built for the wrong interpreter, re-run CMake with
  ``-DPython3_EXECUTABLE="<full path to your venv python>"``.
- If ``cl`` is not found, open **Developer PowerShell for VS 2022** and retry.
- If configure fails due to cached settings, remove ``build/`` and rerun the
  configure command.

Repository pointer page:

- `docs/Install_SALTRO.md <../../Install_SALTRO.md>`_
