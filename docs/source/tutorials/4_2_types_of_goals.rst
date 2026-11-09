Types of Goals
==============

A **three-axis goal**, such as
:class:`~ADCS.CONOPS.goals.attitude_goals.fixed_attitude_goal.Fixed_Attitude_Goal`,
specifies the spacecraft's complete attitude with a reference quaternion.
A **single-axis goal**, such as
:class:`~ADCS.CONOPS.goals.vector_goals.eci_goal.ECI_Goal`, only asks one body direction,
such as a camera boresight, to point along a chosen inertial vector. Rotation
around that direction remains free.

Both examples below start MiRaTa from the same attitude and use the same
four-wheel pyramid, ``MTQ_w_RW`` controller, 480-second run, and inertial-frame
Godot camera. The yellow line is MiRaTa's body boresight; green is its target
direction. Both runs start with a twist around the boresight. In the
three-axis run, the controller rotates around that direction to remove the
twist and reach the specified quaternion. The single-axis goal does not
command that rotation: it only aligns the yellow boresight with the green
target direction.

.. container:: dynamics-visualization-grid

   .. container:: installation-card

      .. rubric:: Three-axis attitude goal

      .. image:: ../_static/tutorials/controlling_spacecraft/three_axis_goal.gif
         :alt: MiRaTa rotates its full attitude toward the fixed quaternion target above Earth.
         :width: 100%

      .. literalinclude:: ../../../examples/tutorials/controlling_spacecraft/three_axis_goal.py
         :language: python
         :caption: three_axis_goal.py

   .. container:: installation-card

      .. rubric:: Single-axis vector goal

      .. image:: ../_static/tutorials/controlling_spacecraft/vector_goal.gif
         :alt: MiRaTa aligns its boresight with an inertial direction while rotation about that direction remains free.
         :width: 100%

      .. literalinclude:: ../../../examples/tutorials/controlling_spacecraft/vector_goal.py
         :language: python
         :caption: vector_goal.py

Both runs align the boresight to within about 1° by the end. The three-axis
run also reaches within about 1° of the complete target attitude; the vector
run retains about 60° of rotation around the boresight.

“Free to rotate around the pointing axis” means that any final angle around
that axis satisfies a single-axis goal. It does not mean that the controller
will deliberately spin the spacecraft around it; this controller also damps
angular velocity.

.. admonition:: Choosing a goal
   :class: tutorial-tip

   A single-axis goal is generally easier to achieve because the spacecraft
   can rotate freely around the aligned axis. Choose a three-axis goal when
   that remaining rotation matters, such as when a solar panel or antenna
   must also face a particular direction.

