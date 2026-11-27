Tutorials
=========

Tutorials to help you get started with Generalized ADCS. They build on each
other, from the physics of a rotating spacecraft to full missions, so working
through them in order is the easiest path. Pages with an example show the
script they run; the scripts live in ``examples/tutorials/`` in the repository
(see :doc:`../getting_started/examples`).

New here? Start with :doc:`0_0_welcome`, and make sure you have
:doc:`installed Generalized ADCS <../installation/index>` and tried the
:doc:`../getting_started/quickstart`.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 1 · Spacecraft Dynamics
      :link: 1_0_spacecraft_dynamics
      :link-type: doc
      :footer: 3 tutorials

      Start with the coordinate frames used to describe motion, then explore
      free rotation and the intermediate-axis flip.

   .. grid-item-card:: 2 · Torques on a Spacecraft
      :link: 2_0_torques_on_a_spacecraft
      :link-type: doc
      :footer: 8 tutorials

      Environmental torques disturb a spacecraft’s attitude. Reaction wheels
      and magnetorquers apply control torques to change it.

   .. grid-item-card:: 3 · Forces on a Spacecraft
      :link: 3_0_forces_on_a_spacecraft
      :link-type: doc
      :footer: 2 tutorials

      Orbital forces include Earth’s nonspherical gravity, atmospheric drag,
      and solar radiation pressure. These examples explore gravity harmonics
      and drag with SGP4.

   .. grid-item-card:: 4 · Controlling Spacecraft
      :link: 4_0_controlling_spacecraft
      :link-type: doc
      :footer: 8 tutorials

      Controllers turn pointing goals into actuator commands. These examples
      cover reaction wheels, magnetorquers, and momentum management.

   .. grid-item-card:: 5 · Attitude Sensors
      :link: 5_0_attitude_sensors
      :link-type: doc
      :footer: 7 tutorials

      Sensors measure motion, orientation, and position. Their readings can
      include noise, bias, and periods when a target is outside the field of
      view.

   .. grid-item-card:: 6 · Estimators
      :link: 6_0_estimators
      :link-type: doc
      :footer: 8 tutorials

      Estimators combine sensor readings with a dynamics model to infer the
      spacecraft state. Start with attitude determination, then explore
      filters and practical measurement limits.

   .. grid-item-card:: 7 · Monte Carlo
      :link: 7_0_monte_carlo
      :link-type: doc
      :footer: 5 tutorials

      Repeat a simulation while varying the orbit, initial attitude, or goal.
      The final example checks controller performance across those runs.

   .. grid-item-card:: 8 · Simulating Your First Mission
      :link: 8_0_simulating_first_mission
      :link-type: doc
      :footer: 3 tutorials

      Follow a mission from deployment through detumbling to ground-target
      pointing.

   .. grid-item-card:: 9 · Advanced Topics
      :link: 9_0_advanced_topics
      :link-type: doc
      :footer: 3 tutorials · some need an optional add-on

      Trajectory planners and running part of a simulation on another
      computer.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   0_0_welcome
   1_0_spacecraft_dynamics
   2_0_torques_on_a_spacecraft
   3_0_forces_on_a_spacecraft
   4_0_controlling_spacecraft
   5_0_attitude_sensors
   6_0_estimators
   7_0_monte_carlo
   8_0_simulating_first_mission
   9_0_advanced_topics

..
   Future tutorials: the following tutorials are planned:

   * 1_4

   (Kept as a comment until the planned tutorial has a title. It used to be
   shown on this page as a bare "1_4" bullet.)
