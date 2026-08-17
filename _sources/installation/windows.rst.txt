Windows (Native) Installation
=============================

Prerequisites
-------------
- C++ toolchain (Visual Studio recommended)
- Python 3.13+
- Python executable available on PATH

Ensure Correct Python Version
-----------------------------
Verify Python version:

.. code-block:: powershell

   python --version

If needed, explicitly invoke Python 3.13:

.. code-block:: powershell

   py -3.13 --version

Virtual Environment
-------------------
From the ``Generalized_ADCS/`` directory:

.. code-block:: powershell

   python -m venv venv
   .\.venv\Scripts\activate

Install Python dependencies:

.. code-block:: powershell

   pip install -r requirements.txt

The ``--no-build-isolation`` flag is required for Cython access during build.
