Linux / WSL Installation
=======================

Prerequisites
-------------
- Linux or Windows Subsystem for Linux (WSL)

WSL Setup
---------
To install WSL on Windows, open PowerShell **as Administrator** and run:

.. code-block:: powershell

   wsl --install -d Ubuntu

After installation, a Linux terminal will open. Enter a username and password.
The password is required for ``sudo`` commands.

A working shell looks like:

.. code-block:: bash

   user@machine:~$

Connect WSL to VS Code
---------------------
Install the `Remote - WSL <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl>`_
extension in VS Code.

Use the blue icon in the bottom-left corner of VS Code and select
``Connect to WSL``.

Filesystem Note
---------------
Projects should be stored **inside the WSL filesystem**, not on the Windows NTFS
filesystem.

Recommended location:

``Linux/Ubuntu/home/<user>/Documents``

Using Windows paths significantly slows down operations such as
``pip install`` and ``cmake``.

Python Environment Setup
------------------------
Ensure Python 3 is installed:

.. code-block:: bash

   python3 --version

Create and activate a virtual environment:

.. code-block:: bash

   python3 -m venv venv
   source venv/bin/activate

Install Python dependencies:

.. code-block:: bash

   pip install -r requirements.txt
   pip install git+https://github.com/jcrudy/choldate.git --no-build-isolation

The ``--no-build-isolation`` flag is required because ``choldate`` depends on
Cython during compilation.
