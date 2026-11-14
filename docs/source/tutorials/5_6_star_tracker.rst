Star Tracker
============

.. image:: ../_static/tutorials/attitude_sensors/star_tracker.gif
   :alt: A rotating spacecraft against the stars in the Godot viewer.
   :width: 90%
   :align: center

A star tracker compares the directions of stars in its camera image with a
catalog of known star directions. In this repo, ``StarTracker`` returns the
body-frame direction of the brightest usable star. ``StarTrackerQuaternion``
uses multiple stars to return a full attitude quaternion. This example uses
the quaternion model and requires at least two usable stars.

The plot compares each reported quaternion with the true simulated attitude.
Blue points show the measurement error. Gray crosses mark times when the
tracker returns ``NaN`` because too few stars are usable. The model filters
stars outside its field of view and stars blocked by Earth or Moon, and
rejects views too close to the Sun.

.. image:: ../_static/tutorials/attitude_sensors/star_tracker.png
   :alt: Star tracker attitude error and times with no solution.
   :width: 100%

The script simulates the tracker, opens the plot, and launches the Godot viewer.

.. literalinclude:: ../../../examples/tutorials/attitude_sensors/star_tracker_godot.py
   :language: python
   :caption: star_tracker_godot.py
