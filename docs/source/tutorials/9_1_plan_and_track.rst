Plan and Track Trajectory Planning
==================================

.. image:: ../_static/tutorials/advanced_topics/plan_and_track.gif
   :alt: Godot animation of MiRaTa following the Plan and Track pointing maneuvers.
   :width: 80%
   :align: center

BeaverCube 1 retargets using three magnetorquers and no reaction wheels.
Plan and Track first optimizes a feasible reference trajectory, then follows
it with time-varying LQR feedback. The example switches between three inertial
pointing goals, with no-goal intervals between them. The animation uses the
MiRaTa model to visualize the BeaverCube 1 simulation.

Install the optional :doc:`Trajectory Planner add-on <../installation/trajectory_planner_addon>`
before running the example.

Planned trajectory
------------------

``ADCS.simulate`` opens the open-loop plan automatically. Its four panels show
body angular rates, magnetorquer commands, the final tracking-error
histogram, and target tracking error.

.. image:: ../_static/tutorials/tutorial_06_planned_trajectory.png
   :alt: Open-loop planned trajectory with angular rates, MTQ commands, final error distribution, and target tracking.
   :width: 90%
   :align: center

Tracked trajectory
------------------

The example opens a second window with the same four panels for the simulated
closed-loop response. It also opens a Godot animation of the spacecraft.

.. image:: ../_static/tutorials/tutorial_06_tracked_trajectory.png
   :alt: Closed-loop tracked trajectory with angular rates, MTQ commands, final error distribution, and target tracking.
   :width: 90%
   :align: center

.. literalinclude:: ../../../examples/tutorials/advanced_topics/plan_and_track.py
   :language: python
   :caption: plan_and_track.py
