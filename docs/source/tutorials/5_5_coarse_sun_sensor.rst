Coarse Sun Sensor
=================

.. image:: ../_static/tutorials/attitude_sensors/coarse_sun_sensor.gif
   :alt: A rotating spacecraft with the Sun visible in the Godot view.
   :width: 90%
   :align: center

A coarse Sun sensor measures the cosine of the angle between its body axis and
the Sun, clipped to zero when the Sun is behind the sensor. This example spins
the spacecraft so that angle changes. The Godot camera starts facing the Sun.
The upper plot shows the recorded sensor reading; the lower plot shows its
angle to the Sun. Shaded periods are eclipse, when the reading is invalid.

.. image:: ../_static/tutorials/attitude_sensors/coarse_sun_sensor.png
   :alt: Recorded coarse Sun sensor reading and angle to the Sun over time, with eclipse shaded.
   :width: 100%

.. admonition:: Run it
   :class: run-it

   From the root of a clone of the repository. The ``examples/`` folder
   is not part of the pip package; see :doc:`../getting_started/examples`.

   .. code-block:: bash

      python examples/tutorials/attitude_sensors/coarse_sun_sensor_godot.py

   Scripts ending in ``_godot.py`` open the 3-D Godot viewer.

.. literalinclude:: ../../../examples/tutorials/attitude_sensors/coarse_sun_sensor_godot.py
   :language: python
   :caption: examples/tutorials/attitude_sensors/coarse_sun_sensor_godot.py
