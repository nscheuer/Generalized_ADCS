Gyroscope
=========

.. image:: ../_static/tutorials/spacecraft_dynamics/dzhanibekov_effect/full_inertia.gif
   :alt: A spacecraft tumbles near its intermediate inertia axis in the Dzhanibekov effect.
   :width: 80%
   :align: center

A gyroscope measures the body's angular velocity :math:`\boldsymbol{\omega}`.
By itself, it cannot determine absolute attitude; integrating its rate only
tracks relative rotation and accumulates drift from bias.

.. image:: ../_static/tutorials/attitude_sensors/gyroscope_rates.png
   :alt: True and measured body rates during a tumble, with the measurement residual showing gyro bias and noise.
   :width: 100%

The example uses a deliberately poor gyro: 0.05°/s white noise per sample
and fixed axis offsets of 0.1–0.2°/s. The offsets do not random-walk
(``std_bias=0``).

.. admonition:: Run it
   :class: run-it

   From the root of a clone of the repository. The ``examples/`` folder
   is not part of the pip package; see :doc:`../getting_started/examples`.

   .. code-block:: bash

      python examples/tutorials/attitude_sensors/gyroscope.py

.. literalinclude:: ../../../examples/tutorials/attitude_sensors/gyroscope.py
   :language: python
   :caption: examples/tutorials/attitude_sensors/gyroscope.py

The animation uses the same torque-free intermediate-axis tumble as the
:doc:`Dzhanibekov effect tutorial <1_3_dzhanibekov_effect>`.
