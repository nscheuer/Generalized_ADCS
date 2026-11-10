Sun Sensor
==========

.. image:: ../_static/tutorials/attitude_sensors/sun_sensor.gif
   :alt: Godot view of the Sun vector, Sun sensor field of view, and eclipse timeline.
   :width: 90%
   :align: center

A coarse Sun sensor measures light along its body axis. The Godot view shows
the Sun direction, sensor field of view, and eclipse; the plot shows the
measurement becoming invalid in shadow.

.. image:: ../_static/tutorials/attitude_sensors/sun_sensor_eclipse.png
   :alt: Normalized Sun sensor reading becomes invalid during eclipse.
   :width: 100%

.. literalinclude:: ../../../examples/tutorials/attitude_sensors/sun_sensor.py
   :language: python
   :caption: sun_sensor.py
