SALTRO Add-On
==============

``saltro_py`` is an optional add-on module for
`SALTRO <https://nscheuer.github.io/SALTRO>`__-based planning in
Generalized ADCS.

The C++ source lives in the separate repository
`nscheuer/SALTRO <https://github.com/nscheuer/SALTRO>`__, which Generalized
ADCS pulls in as a git submodule at ``SALTRO/``. Generalized ADCS imports
``saltro_py`` from ``SALTRO/build/``.

Start with :doc:`addon_prerequisites`, then follow the page for your platform:

.. toctree::
   :maxdepth: 1

   saltro_linux
   saltro_windows
   saltro_macos

Canonical SALTRO references:

- `SALTRO documentation website <https://nscheuer.github.io/SALTRO>`__
- `Official SALTRO installation page <https://nscheuer.github.io/SALTRO/datasheets/Installation.html>`__

If you cloned without submodules, initialise SALTRO with:

.. code-block:: bash

   git submodule update --init --recursive SALTRO

Re-run this after pulling new commits if the submodule pointer changed.

Once it is built, run the :doc:`../tutorials/9_2_saltro` tutorial from the
repository root:

.. code-block:: bash

   python examples/tutorials/advanced_topics/saltro_godot.py

See also :doc:`release note 0.1.0 <../release_notes/0_1_0_saltro>` for a quick
comparison of trajectory optimization benefits and drawbacks.
