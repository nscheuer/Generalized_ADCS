Controlling Spacecraft
======================

Controllers turn an attitude goal into actuator commands. The first example
uses reaction wheels to reduce attitude and angular-rate error.

PD Control
----------

The :doc:`reaction wheel PD tutorial <4_1_reaction_wheel_pd>` uses proportional
and derivative feedback to point a spacecraft toward a fixed inertial attitude.

.. toctree::
   :maxdepth: 1

   4_1_reaction_wheel_pd

Types of Goals
--------------

The :doc:`types of goals tutorial <4_2_types_of_goals>` compares a complete
three-axis attitude target with a single-axis pointing target.

.. toctree::
   :maxdepth: 1

   4_2_types_of_goals

Types of Goals II
-----------------

The :doc:`second goals tutorial <4_3_types_of_goals_ii>` shows how to track
a ground coordinate or a direction that changes with the orbit.

.. toctree::
   :maxdepth: 1

   4_3_types_of_goals_ii

GoalList
--------

The :doc:`GoalList tutorial <4_4_goal_list>` schedules several pointing goals
for one controller to follow in sequence.

.. toctree::
   :maxdepth: 1

   4_4_goal_list

Magnetorquer-Only Control
-------------------------

The :doc:`magnetorquer-only tutorial <4_5_magnetorquer_only>` uses Lovera's
control law to reduce pointing error over several orbits.

.. toctree::
   :maxdepth: 1

   4_5_magnetorquer_only

Underactuated Control
---------------------

The :doc:`3+1 tutorial <4_6_underactuated_3plus1>` tracks a ground target
with three magnetorquers and one reaction wheel.

.. toctree::
   :maxdepth: 1

   4_6_underactuated_3plus1
