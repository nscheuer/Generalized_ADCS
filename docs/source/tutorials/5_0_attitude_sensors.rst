Attitude Sensors
================

Sensor and actuator errors shape the measurements and commands used by an
attitude control system. These pages show how to configure white noise and
bias models and how their magnitudes affect the signals.

Noise
-----

See :doc:`Noise <5_1_noise>` for per-sample Gaussian sensor and actuator
noise configuration and plots for different :math:`\sigma` values.

Bias
----

See :doc:`Bias <5_2_bias>` for fixed offsets, random-walk drift, and plots
showing the effect of :math:`\sigma_b`.

Gyroscope
---------

See :doc:`Gyroscope <5_3_gyroscope>` for body-rate measurement, bias, and
noise during a tumbling motion.

Magnetometer
------------

See :doc:`Magnetometer <5_4_magnetometer>` to compare the magnetic field
seen by a stable satellite and a tumbling satellite.

Sun Sensor
----------

See :doc:`Sun Sensor <5_5_sun_sensor>` for the sensor field of view, its
light-dependent reading, and how eclipse invalidates the measurement.

.. toctree::
   :maxdepth: 1
   :hidden:

   5_1_noise
   5_2_bias
   5_3_gyroscope
   5_4_magnetometer
   5_5_sun_sensor
