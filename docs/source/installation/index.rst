Installation
============

Generalized_ADCS supports Linux, WSL, and native Windows installations

Choose the guide appropriate for your system:

.. toctree::
   :maxdepth: 1
   :caption: Installation Guides

   linux
   windows

Furthermore, there is the experimental trajectory planner module, which is compiled separately:

.. toctree::
   :maxdepth: 1
   :caption: Trajectory Planner (Experimental)

   trajectory_planner_linux
   trajectory_planner_windows

Canonical trajectory planner setup (Windows + Linux) is documented in:

- `docs/Install_Trajectory_Planner.md <../../Install_Trajectory_Planner.md>`_

The SALTRO module (``saltro_py``) is also optional and must be built separately
if you want to use SALTRO-based planning.

For SALTRO details and installation:

- `SALTRO documentation website <https://nscheuer.github.io/SALTRO>`_
- `docs/Install_SALTRO.md <../../Install_SALTRO.md>`_