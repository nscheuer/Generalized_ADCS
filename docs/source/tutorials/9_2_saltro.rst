SALTRO Trajectory Planning
==========================

.. image:: ../_static/tutorials/advanced_topics/saltro.gif
   :alt: Godot animation of MiRaTa following a SALTRO planned pointing maneuver.
   :width: 80%
   :align: center

SALTRO plans a constrained attitude trajectory and tracks it with feedback.
The BeaverCube 2 dynamics model uses three magnetorquers and one reaction
wheel. The Godot animation uses the MiRaTa spacecraft model to show its motion.
The tutorial uses the original default SALTRO planner settings for this 3+1
maneuver. In the regenerated run shown here, the open-loop final pointing error
is 0.196° and the closed-loop final error is 0.767°. The final body-rate
magnitude is 0.000641 rad/s. Both plot images and the animation come from that
same run.

Install the optional :doc:`SALTRO add-on <../installation/saltro_addon>` before
running the example. The installation page links to Linux and Windows setup.

Planned trajectory
------------------

``ADCS.simulate`` opens the open-loop planned trajectory. It shows angular
rates, magnetorquer commands, final tracking error, and target tracking.

.. image:: ../_static/tutorials/advanced_topics/saltro_planned.png
   :alt: SALTRO open-loop planned trajectory in four plots.
   :width: 90%
   :align: center

Tracked trajectory
------------------

The example opens the closed-loop response in a second four-panel window.

.. image:: ../_static/tutorials/advanced_topics/saltro_tracked.png
   :alt: SALTRO closed-loop tracked trajectory in four plots.
   :width: 90%
   :align: center

.. literalinclude:: ../../../examples/tutorials/advanced_topics/saltro.py
   :language: python
   :caption: saltro.py
