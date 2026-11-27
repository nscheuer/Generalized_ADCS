EKF with Disturbance Torque Estimation
======================================

This example uses the same noisy sensors and gyro bias estimator as the
previous page. It replaces the residual magnetic dipole with a constant
three-axis torque in the satellite's body frame. The augmented MEKF begins
with a zero torque estimate and updates the three torque components from the
measured motion.

The plots show attitude angle error, gyro bias, and the true and estimated
body torque. The true values are available only because this is a simulation.

.. image:: ../_static/tutorials/estimators/ekf_with_disturbance_torque_estimation.png
   :alt: Attitude angle error, gyro bias estimates, and body torque estimates.
   :width: 100%

.. admonition:: Run it
   :class: run-it

   From the root of a clone of the repository. The ``examples/`` folder
   is not part of the pip package; see :doc:`../getting_started/examples`.

   .. code-block:: bash

      python examples/tutorials/estimators/ekf_with_disturbance_torque_estimation.py

.. literalinclude:: ../../../examples/tutorials/estimators/ekf_with_disturbance_torque_estimation.py
   :language: python
   :caption: examples/tutorials/estimators/ekf_with_disturbance_torque_estimation.py
