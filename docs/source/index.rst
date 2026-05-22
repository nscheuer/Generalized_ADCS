.. Generalized_ADCS documentation master file

Generalized Attitude Determination and Control System
=====================================================

A Python-first framework for simulating spacecraft attitude dynamics,
estimation, and control in orbit.

**Generalized ADCS** is designed for aerospace students, researchers,
and engineers who want a flexible, transparent ADCS simulation
environment without black-box solvers. The project repository is
available at `nscheuer/Generalized_ADCS <https://github.com/nscheuer/Generalized_ADCS>`_,
and it is developed with `MIT STAR Lab <https://aeroastro.mit.edu/starlab/>`_.

Latest Release Notes
====================

.. container:: featured-release-box

   .. image:: _static/release_notes/0_1_0_saltro_live.gif
      :alt: Trajectory optimization animation
      :width: 300px
      :align: right

   **The age of trajectory optimization is upon us!**

   Generalized_ADCS now includes two optional add-ons:
   :doc:`Trajectory Planner <installation/trajectory_planner_addon>` and :doc:`SALTRO <installation/saltro_addon>`.
   Trajectory optimization enables more aggressive retargeting maneuvers than simple point-to-point commands.

   :doc:`→ Explore release note 0.1.0 <release_notes/0_1_0_saltro>`

What you can do with Generalized ADCS
-------------------------------------
- Simulate rigid-body spacecraft attitude dynamics
- Combine reaction wheels, magnetorquers, and sensors
- Implement and test custom controllers and estimators
- Run closed-loop simulations using orbital states
- Visualize pointing performance and control effort

.. code-block:: python

   import ADCS as ADCS
   import numpy as np
   import matplotlib.pyplot as plt

   acts = [ADCS.RW(axis=np.array([1, 0, 0]), max_torque=0.0023, J=5.7e-6, h=0.0, h_max=0.0036)]
   acts += [ADCS.MTQ(axis, max_torque=0.2) for axis in np.eye(3)]
   sens = [ADCS.MTM(axis) for axis in np.eye(3)]

   satellite = ADCS.Satellite(mass=10, J_0=np.diag([0.03, 0.03, 0.01]), actuators=acts, sensors=sens, boresight=np.array([0, 0, 1]))
   x_0 = np.array([0.01, -0.02, 0.01] + [1, 0, 0, 0] + [0.0])  # w, q, h

   controller = ADCS.controller.MTQ_w_RW_LP(est_sat=satellite, p_gain=0.00005, d_gain=0.002, c_gain=0.001, h_target=np.array([0, 0, 0]))

   goal = ADCS.goals.Coordinate_Goal(lat=42.36, lon=-71.06, alt=0)
   os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(),J2000=0.22, R=np.array([5000, 0, 5000]), V=np.array([0, 7.5, 0]))

   results = ADCS.simulate(
      x=x_0,
      satellite=satellite,
      controller=controller,
      goal=goal,
      os0=os0,
      dt=1.0,
      tf=500.0
   )

   ADCS.plot(
      results,
      ADCS.plots.AnimationPlot(goal=goal),
      layout=(1,1),
      title="Underactuated Control Animation",
   )

   plt.show()
   

.. image:: _static/boston_tracking.png
   :alt: Simulation of a satellite in orbit with a ground pointing target.
   :width: 400px
   :align: center

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation/index
   tutorials/index
   release_notes/index

.. toctree::
   :maxdepth: 1
   :caption: Function Documentation

   ADCS

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing/index
   contributing/testing
   contributing/documentation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
