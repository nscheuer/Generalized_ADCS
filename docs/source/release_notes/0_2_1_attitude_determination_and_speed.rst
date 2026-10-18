0.2.1 Attitude Determination and Performance
============================================

Single-frame attitude determination
-----------------------------------

TRIAD, QUEST, and Davenport's q-method now determine attitude from direction
observations in a single frame. Estimators can use these solutions to initialize
from their first sensor readings through :class:`~ADCS.estimators.initialization.WarmStart`.
The warm start also derives angular rate and its initial covariance from the
gyros. Select a solver with ``WarmStart(method="triad")``, ``"quest"``, or
``"q_method"``; ``"auto"`` uses a quaternion star tracker when available and
QUEST otherwise.

Geomagnetic field and disturbance fixes
---------------------------------------

A faster IGRF geomagnetic field implementation is available for orbit and
sensor calculations. True disturbances now advance once per plant step, so
time-varying disturbance models evolve during simulation as intended.

Performance and tests
---------------------

Math operations, estimator calculations, and orbit tests have been sped up.
The test suite has also been reorganized by subsystem and behavior.

Comparison examples
-------------------

The MEKF is generally faster than the UKF because it propagates one estimate
and linearizes the dynamics and measurement model, while the UKF evaluates a
set of sigma points. That linearization is local, so a large initial attitude
error can make the MEKF fail to converge. Initializing the attitude from the
first sensor readings can move the estimate into a region where its local
corrections work well, as the example below shows.

.. image:: ../_static/release_notes/0_2_1_mekf_warm_start.png
   :alt: MEKF attitude error with a 170 degree initial error and with TRIAD, QUEST, and q-method warm starts.
   :width: 800px
   :align: center

The `MEKF warm-start comparison script <https://github.com/nscheuer/Generalized_ADCS/blob/main/examples/estimators/example_compare_mekf_warm_starts.py>`_
produces this plot.
