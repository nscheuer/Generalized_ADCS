0.1.0 The age of trajectory optimization is upon us! (2026-04-10)
===================================================================

Generalized_ADCS now includes two optional trajectory-optimization add-ons:

- :doc:`Trajectory Planner Add-On <../installation/trajectory_planner_addon>`
- :doc:`SALTRO Add-On <../installation/saltro_addon>`

What Is Trajectory Optimization?
--------------------------------
Trajectory optimization computes a control-aware attitude trajectory that satisfies
actuator limits and pointing goals over time. In practice, this enables more
aggressive retargeting maneuvers than simple point-to-point attitude commands.

Live Trajectory Optimization Preview
------------------------------------
The animation below shows a live trajectory-optimization maneuver.

.. image:: ../_static/release_notes/0_1_0_saltro_live.gif
   :alt: Live trajectory optimization animation.
   :width: 700px
   :align: center

Quick Tradeoffs
---------------

.. list-table:: Trajectory Optimization Tradeoffs
   :widths: 20 40 40
   :header-rows: 1

   * - Signal
     - Benefits
     - Drawbacks
   * - 🟢
     - Better pointing for difficult retargeting and constrained actuation
     - Higher onboard/ground compute demand
   * - 🔴
     - Cleaner handling of constraints and multi-phase goals
     - More parameters to tune and validate before flight use

Learn More
----------
- :doc:`Tutorial 06: Trajectory Planner (Plan-and-Track) <../tutorials/06_trajectory_planner>`
- :doc:`Tutorial 07: SALTRO Planner <../tutorials/07_SALTRO>`
- `SALTRO documentation website <https://nscheuer.github.io/SALTRO>`_
