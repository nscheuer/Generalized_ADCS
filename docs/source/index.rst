.. Generalized_ADCS documentation master file

Generalized Attitude Determination and Control System
=====================================================

A Python-first framework for simulating spacecraft attitude dynamics,
estimation, and control in orbit.

**Generalized ADCS** is designed for aerospace students, researchers,
and engineers who want a flexible, transparent ADCS simulation
environment without black-box solvers.

Latest Release Notes
====================

.. raw:: html

   <div class="release-grid">
     <a class="release-card release-card-featured" href="release_notes/0_1_6_benchmark.html">
       <img src="_static/release_notes/0_1_6_benchmark_small.png" alt="Benchmark report preview">
       <div class="release-card-copy">
         <div class="release-card-kicker">Update 0.1.6</div>
         <h3>Benchmark</h3>
         <p>New performance benchmarks run on every pull request into <code>main</code> and compare results to catch regressions early.</p>
         <span>Explore release note 0.1.6</span>
       </div>
     </a>
     <a class="release-card" href="release_notes/0_1_5_bugfixes_and_optimizations.html">
       <div class="release-card-copy">
         <div class="release-card-kicker">Update 0.1.5</div>
         <h3>Bugfixes and Optimizations</h3>
         <p>Quality-of-life improvements, bug fixes, and test coverage expansions across the codebase.</p>
         <span>Explore release note 0.1.5</span>
       </div>
     </a>
     <a class="release-card" href="release_notes/0_1_0_saltro.html">
       <img src="_static/release_notes/0_1_0_saltro_live.gif" alt="Trajectory optimization animation">
       <div class="release-card-copy">
         <div class="release-card-kicker">Update 0.1.0</div>
         <h3>The age of trajectory optimization</h3>
         <p>Trajectory Planner and SALTRO introduced aggressive retargeting maneuvers beyond simple point-to-point commands.</p>
         <span>Explore release note 0.1.0</span>
       </div>
     </a>
   </div>

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
