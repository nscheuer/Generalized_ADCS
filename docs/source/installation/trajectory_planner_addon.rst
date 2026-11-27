Trajectory Planner Add-On
=========================

``trajectory_planner`` is an optional optimization add-on for hard pointing
maneuvers in Generalized ADCS. It computes a reference attitude and control
trajectory, which ADCS controllers then track.

The C++ source lives in a separate repository,
`patrickmckeen/OldPlanner <https://github.com/patrickmckeen/OldPlanner>`__,
which Generalized ADCS pulls in as a git submodule at ``OldPlanner/``.
Generalized ADCS imports the resulting ``tplaunch`` / ``pysat`` modules via
``ADCS.controller.helpers.optional_dependencies.get_trajectory_planner_modules()``,
which expects them to be built into ``OldPlanner/build/``.

For algorithm details and design background, see the
`SALTRO documentation website <https://nscheuer.github.io/SALTRO>`__.

Start with :doc:`addon_prerequisites` (submodules and the Python environment),
then follow the page for your platform:

.. toctree::
   :maxdepth: 1

   trajectory_planner_linux
   trajectory_planner_windows
   trajectory_planner_macos

Run the trajectory planner examples
-----------------------------------

Run from the repository root with your virtual environment active:

.. code-block:: bash

   python examples/tutorials/advanced_topics/plan_and_track_godot.py
   python examples/tutorials/advanced_topics/saltro_godot.py

The first script is the :doc:`../tutorials/9_1_plan_and_track` tutorial. The
second is the :doc:`../tutorials/9_2_saltro` tutorial and needs the
:doc:`SALTRO add-on <saltro_addon>` instead. The controller examples in
``examples/controllers/example_altro_*.py`` also use this add-on.

See also :doc:`release note 0.1.0 <../release_notes/0_1_0_saltro>` for a concise
overview of trajectory optimization and add-on selection.
