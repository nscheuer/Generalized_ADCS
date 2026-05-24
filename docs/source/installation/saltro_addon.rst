SALTRO Add-On
==============

``saltro_py`` is an optional add-on module for `SALTRO <https://nscheuer.github.io/SALTRO>`_-based planning in
Generalized_ADCS.
Its source is pulled in via the ``SALTRO/`` git submodule.

.. toctree::
   :maxdepth: 1

   saltro_linux
   saltro_windows

Canonical SALTRO references:

- `SALTRO documentation website <https://nscheuer.github.io/SALTRO>`_
- `docs/Install_SALTRO.md <../../Install_SALTRO.md>`_

If you cloned without submodules, initialise SALTRO with:

.. code-block:: bash

   git submodule update --init --recursive SALTRO

See also :doc:`release note 0.1.0 <../release_notes/0_1_0_saltro>` for a quick
comparison of trajectory optimization benefits and drawbacks.
