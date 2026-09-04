A Feasibility-Aware Attitude Trajectory Planner for Small Satellites
====================================================================

Patrick McKeen\ :sup:`1`, Niclas Scheuer\ :sup:`2`, Kerri Cahoy\ :sup:`1`

:sup:`1` Massachusetts Institute of Technology ·
:sup:`2` ETH Zurich

Flash Talks Session 3, 40th Annual Small Satellite Conference, Salt Lake City,
Utah, August 25, 2026.

Read it
-------

- **Paper page** — https://digitalcommons.usu.edu/smallsat/2026/all2026/227/
- **Proceedings page** — https://digitalcommons.usu.edu/smallsat/2026/all2026/227/
- **Contact the authors** — :doc:`../ssc26/contact`

Abstract
--------

Spacecraft attitude control traditionally requires extensive per-mission
engineering--hand-tuned gains, mission-specific mode logic, and conservative
operating envelopes--that breaks down when on-orbit conditions deviate from
ground assumptions. This is especially costly for small satellites, where
limited ground support makes autonomous adaptability essential. We present a
feasibility-aware attitude trajectory planner that replaces this
hand-engineered stack with continuous planning from a digital twin: less a new
feedback law than a different division of labor, in which the mode logic, gain
tuning, and desaturation a conventional ADCS builds as separate pieces are
instead absorbed into a single optimization.

Built on the ALTRO trajectory optimization algorithm, the planner accepts
high-level pointing goals and reasons over future time-varying control
authority, respecting the full nonlinear dynamics and actuator constraints
directly--without collocation or slack variables. Because trajectories are
produced by forward integration, every returned trajectory is dynamically
consistent with the modeled spacecraft; when a goal cannot be met within the
modeled constraints, the planner returns a bounded, best-effort trajectory
rather than exceeding actuator authority or diverging. It also emits
time-varying LQR gains for closed-loop tracking, providing robustness to
unmodeled disturbances, estimation error, and sensor noise without anticipating
them at the planning stage.

Optimizing directly against the dynamics, the planner also finds strategies a
designer would otherwise have to anticipate: facing a body-fixed disturbance
beyond its actuator authority, it autonomously adopted a spinning maneuver that
averages the disturbance out over each revolution--a solution it was never
directed toward. Monte Carlo simulations (100 runs, 1000s, randomized
orientation, goal, rate, and orbit, under disturbances excluded from the
planner's model) show 84% of slews converging (< 5 degrees, mean 3.8 degrees)
for a magnetorquer-only spacecraft versus 27% (mean 21.5 degrees) for PD
control, improving to 94% (mean 1.2 degrees) versus 90% (mean 9.1 degrees) with
a single reaction wheel added.

The same framework spans magnetorquer-only, hybrid, and full reaction-wheel
configurations with minimal reconfiguration, handling momentum management and
desaturation implicitly within slews and pointing. Implemented in C++ for
embedded-class processors, it replans continuously on overlapping
horizons--bringing trajectory-planned autonomy to spacecraft historically
limited to reactive control.

Cite it
-------

.. code-block:: bibtex

   @inproceedings{mckeen2026feasibility,
     title     = {A Feasibility-Aware Attitude Trajectory Planner for Small Satellites},
     author    = {McKeen, Patrick and Scheuer, Niclas and Cahoy, Kerri},
     booktitle = {Proceedings of the 40th Annual Small Satellite Conference},
     year      = {2026},
     address   = {Salt Lake City, UT},
     url       = {https://digitalcommons.usu.edu/smallsat/2026/all2026/227/},
   }
