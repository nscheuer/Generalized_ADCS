Setting Up WSL
==============

Windows Subsystem for Linux (WSL) lets you follow the **Linux / WSL**
instructions on a Windows machine, including the optional C++ add-on builds.

Install WSL
-----------

Open PowerShell in Administrator mode and run:

.. code-block:: powershell

   wsl --install -d Ubuntu

After installation, Windows opens a new terminal window. Enter a ``username``
and ``password``. Remember the password: it is required to run ``sudo``
commands in the WSL shell. When this is complete you have a working Linux
shell:

.. code-block:: text

   user@machine:~$

Check that Python 3.10 or newer is installed in your WSL distribution with
``python3 --version``, then continue with the **Linux / WSL** tab on the
:doc:`Installation <index>` page.

Connect WSL to VS Code
----------------------

Open VS Code in Windows and install the
`Remote-WSL <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl>`__
extension. You can then open WSL in VS Code by selecting the blue arrow button
in the bottom left of the window and choosing **Connect to WSL**.

Keep projects on the Linux filesystem
-------------------------------------

Store and clone projects on the WSL partition of your PC rather than in the
Windows filesystem. In File Explorer, a **Linux** folder appears at the bottom
of the left pane; a good location for projects is
``Linux/Ubuntu/home/<user>/Documents``.

Working on a Windows path from WSL is very slow: Windows uses an NTFS
filesystem while WSL works with ext4 images, with a virtualization layer in
between. That layer can make operations such as ``pip install ...`` and
``cmake ...`` extremely slow.

Debugging tools
---------------

To debug the C++ add-ons under WSL, install GDB:

.. code-block:: bash

   sudo apt update
   sudo apt install gdb
   which gdb
