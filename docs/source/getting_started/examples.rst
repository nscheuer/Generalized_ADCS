Running the Examples
====================

Generalized ADCS has two sets of examples:

- **Bundled examples** ship inside the Python package, so they work right after
  ``pip install generalized-adcs``.
- **The full collection** lives in the repository's top-level
  `examples/ <https://github.com/nscheuer/Generalized_ADCS/tree/main/examples>`__
  directory. It holds every tutorial script plus controller, estimator, Monte
  Carlo, remote-execution and save/load examples. It is **not** part of the pip
  package, so you need the files from GitHub.

Bundled examples
----------------

Run these from any environment where the package is installed:

.. tab-set::

   .. tab-item:: basic_simulation

      Propagate a freely rotating spacecraft for 30 seconds
      (walked through in the :doc:`quickstart`).

      .. code-block:: bash

         python -m ADCS.examples.basic_simulation

      .. code-block:: text

         Final angular velocity [rad/s]: [0.         0.         0.01745329]
         Final attitude quaternion [scalar, x, y, z]: [0.96814764 0.         0.         0.25038   ]

   .. tab-item:: orbit_elements

      Create an orbital state and print its Keplerian elements.

      .. code-block:: bash

         python -m ADCS.examples.orbit_elements

      .. code-block:: text

         Semi-major axis: 7172.78 km
         Eccentricity: 0.191286
         Inclination: 11.66 deg
         RAAN: 8.13 deg
         Argument of periapsis: 263.17 deg
         True anomaly: 96.83 deg

      .. literalinclude:: ../../../ADCS/examples/orbit_elements.py
         :language: python

   .. tab-item:: environment_field

      Evaluate the default IGRF-14 magnetic field at one ECEF position.

      .. code-block:: bash

         python -m ADCS.examples.environment_field

      .. code-block:: text

         Magnetic field at 400 km altitude [T]: [ 1.17307659e-05 -1.73393680e-06  2.26483523e-05]

      .. literalinclude:: ../../../ADCS/examples/environment_field.py
         :language: python

Get the full collection
-----------------------

.. tab-set::

   .. tab-item:: Clone the repository (recommended)

      Cloning also gives you the debug scripts, the paper pipelines, the tests
      and the documentation source:

      .. code-block:: bash

         git clone --depth 1 https://github.com/nscheuer/Generalized_ADCS.git
         cd Generalized_ADCS

      To run the examples against this checkout instead of a PyPI install,
      install it in editable mode (see :doc:`../installation/index`).

   .. tab-item:: Download only examples/

      GitHub does not provide a direct ``curl`` download for one folder. The
      command below uses GitHub's tree API to find the files under
      ``examples/`` and ``curl`` to download them. It requires Python 3 and
      ``curl``:

      .. code-block:: bash

         mkdir -p examples
         curl -fsSL "https://api.github.com/repos/nscheuer/Generalized_ADCS/git/trees/main?recursive=1" |
         python3 -c 'import json, pathlib, subprocess, sys; [ (pathlib.Path(f["path"]).parent.mkdir(parents=True, exist_ok=True), subprocess.run(["curl", "-fsSL", "https://raw.githubusercontent.com/nscheuer/Generalized_ADCS/main/" + f["path"], "-o", f["path"]], check=True)) for f in json.load(sys.stdin)["tree"] if f["type"] == "blob" and f["path"].startswith("examples/") ]'

Run a script
------------

Run scripts from the repository root by path. For example, the script behind
the :doc:`../tutorials/3_1_j2_perturbation` tutorial:

.. code-block:: bash

   python examples/tutorials/forces_on_a_spacecraft/j2_perturbation.py

Most scripts open matplotlib figure windows when they finish.

.. tip::

   Scripts whose names end in ``_godot.py`` open the 3-D Godot viewer. The
   first time it opens, the viewer offers to download the Godot engine for
   your operating system (asking in the terminal first) and keeps it in a
   per-user cache. Planet textures and the optional NASA spacecraft models are
   also downloaded only after a terminal prompt; declining leaves the viewer
   running with built-in colours. Tutorial scripts that open the Godot viewer
   leave spacecraft-model selection to you: use the model menu in the viewer
   toolbar to choose the simulation geometry, a box, or an available
   spacecraft model. To use another Godot build, pass ``binary=`` or set the
   ``ADCS_VIEWER_BIN`` environment variable.

What's in examples/
-------------------

``examples/tutorials/``
   One folder per tutorial section. Each script accompanies
   the tutorial page of the same name.

   .. list-table::
      :header-rows: 1
      :widths: 35 65

      * - Folder
        - Tutorial section
      * - ``spacecraft_dynamics/``
        - :doc:`../tutorials/1_0_spacecraft_dynamics`
      * - ``torques_on_a_spacecraft/``
        - :doc:`../tutorials/2_0_torques_on_a_spacecraft`
      * - ``forces_on_a_spacecraft/``
        - :doc:`../tutorials/3_0_forces_on_a_spacecraft`
      * - ``controlling_spacecraft/``
        - :doc:`../tutorials/4_0_controlling_spacecraft`
      * - ``attitude_sensors/``
        - :doc:`../tutorials/5_0_attitude_sensors`
      * - ``estimators/``
        - :doc:`../tutorials/6_0_estimators`
      * - ``monte_carlo/``
        - :doc:`../tutorials/7_0_monte_carlo`
      * - ``first_mission/``
        - :doc:`../tutorials/8_0_simulating_first_mission`
      * - ``advanced_topics/``
        - :doc:`../tutorials/9_0_advanced_topics`

   The folder also keeps five scripts from an earlier tutorial series:
   ``tutorial_1_underactuated_control.py``, ``tutorial_2_noisy_control.py``,
   ``tutorial_3_simple_estimation_augmented.py``,
   ``tutorial_4_complex_estimation_augmented.py`` and
   ``tutorial_5_orbit_estimation.py``.

``examples/controllers/``
   One script per controller and goal type. The suffix selects the goal:
   ``_full`` points to a fixed attitude (``Fixed_Attitude_Goal``),
   ``_reduced`` points one axis along an inertial vector (``ECI_Goal``), and
   ``_mixed`` uses a ``GoalList`` that switches from a ground coordinate to
   anti-Sun pointing at 500 s.

   - ``example_altro_{full,mixed,reduced}.py``
   - ``example_bdot.py``
   - ``example_lovera_{full,mixed,reduced}.py``
   - ``example_lp_{full,mixed,reduced}.py``
   - ``example_mtq_w_rw_{full,mixed,reduced}.py`` and ``example_mtq_w_rw_coordinate.py``
   - ``example_qpw_{full,mixed,reduced}.py``
   - ``example_wisniewski_{full,mixed,reduced}.py``

   The ALTRO examples need the optional
   :doc:`trajectory planner add-on <../installation/trajectory_planner_addon>`.
   The controllers themselves are described in the :doc:`../ADCS` and, with
   their source papers, on the :doc:`../contributing/citations` page.

``examples/estimators/``
   ``example_ekf.py``, ``example_mekf.py``, ``example_ukf.py`` and
   ``example_srukf.py``, plus ``example_advanced_*.py`` versions of each.
   They are introduced in :doc:`release note 0.2.0
   <../release_notes/0_2_0_estimation_framework>`; the
   :doc:`../tutorials/6_0_estimators` tutorials build up to them.

``examples/mc/``
   ``example_single_controller.py`` runs one simulation with ``ADCS.simulate``;
   ``example_mc_controller.py`` runs a Monte Carlo ensemble with
   ``ADCS.simulate_mc``. See :doc:`../tutorials/7_0_monte_carlo`.

``examples/remote/``
   ``run_remote_universal_augmented.py`` starts a worker on a second computer
   that computes one component, such as the controller commands, while another
   computer propagates the simulation. See
   :doc:`../tutorials/9_3_remote_execution`.

``examples/save_and_load/``
   ``example_save_results.py`` simulates and saves the results with
   ``results.save(...)``, which writes a timestamped
   ``examples/output/example_save_results_<date>_<time>.sim`` file.
   ``example_load_results.py`` reads one back with
   ``ADCS.SimulationResults.load(...)``; edit its path to point at a ``.sim``
   file you saved, because the file it names is not in the repository.

``examples/release_notes/``
   The examples from :doc:`release note 0.0.1 <../release_notes/0_0_1_mc>`
   (``example_0_0_1_multi_mc.py``, ``example_0_0_1_reduced.py``) and
   :doc:`release note 0.0.2 <../release_notes/0_0_2_multi_boresight>`
   (``example_0_0_2_multi_boresight.py``).
