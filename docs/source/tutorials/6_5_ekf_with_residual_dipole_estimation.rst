EKF with Residual Dipole Estimation
===================================

This example keeps the noisy sensors and gyro bias estimation from the
previous page. The satellite now has a fixed residual magnetic dipole. Its
interaction with Earth's magnetic field produces a torque that changes the
satellite's motion. The augmented MEKF starts with a zero dipole estimate and
updates its three components alongside attitude and gyro bias.

The plots show attitude angle error, true versus estimated gyro bias, and
true versus estimated residual dipole. The true values are available only
because this is a simulation.

.. image:: ../_static/tutorials/estimators/ekf_with_residual_dipole_estimation.png
   :alt: Attitude angle error, gyro bias estimates, and residual dipole estimates.
   :width: 100%

.. admonition:: Run it
   :class: run-it

   From the root of a clone of the repository. The ``examples/`` folder
   is not part of the pip package; see :doc:`../getting_started/examples`.

   .. code-block:: bash

      python examples/tutorials/estimators/ekf_with_residual_dipole_estimation.py

.. literalinclude:: ../../../examples/tutorials/estimators/ekf_with_residual_dipole_estimation.py
   :language: python
   :caption: examples/tutorials/estimators/ekf_with_residual_dipole_estimation.py
