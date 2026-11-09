Types of Goals II
=================

The previous :doc:`goal comparison <4_2_types_of_goals>` used fixed inertial
targets. These two goals move as the spacecraft orbits Earth. A
**coordinate goal** (:class:`~ADCS.CONOPS.goals.vector_goals.coordinate_goal.Coordinate_Goal`)
tracks a location fixed on the rotating Earth; a **nadir goal**
(:class:`~ADCS.CONOPS.goals.vector_goals.nadir_goal.Nadir_Goal`)
tracks the direction toward Earth's center. Both are
single-axis goals: they align MiRaTa's body boresight and leave rotation
around it free.

Both animations use the same MiRaTa satellite, controller, starting state,
and inertial camera as the previous comparison. Yellow marks the body
boresight; green marks the current target direction.

.. container:: dynamics-visualization-grid

   .. container:: installation-card

      .. rubric:: Ground coordinate

      .. image:: ../_static/tutorials/controlling_spacecraft/coordinate_goal.gif
         :alt: MiRaTa tracks a fixed ground location as it flies over Earth.
         :width: 100%

      .. literalinclude:: ../../../examples/tutorials/controlling_spacecraft/coordinate_goal.py
         :language: python
         :caption: coordinate_goal.py

   .. container:: installation-card

      .. rubric:: Nadir tracking

      .. image:: ../_static/tutorials/controlling_spacecraft/nadir_goal.gif
         :alt: MiRaTa keeps its boresight aimed toward Earth's center.
         :width: 100%

      .. literalinclude:: ../../../examples/tutorials/controlling_spacecraft/nadir_goal.py
         :language: python
         :caption: nadir_goal.py

The coordinate goal takes ``lat``, ``lon``, and ``alt``: geodetic latitude and longitude in
degrees and altitude in kilometers. Its target rotates with Earth, so the
line of sight changes as both Earth and the spacecraft move. The nadir goal
uses the spacecraft's current position, so its target continuously follows
the orbit. Each script includes its own satellite and controller setup and
can be run independently.

Other orbit-related direction goals include
:class:`~ADCS.CONOPS.goals.vector_goals.zentih_goal.Zenith_Goal` (away from Earth),
:class:`~ADCS.CONOPS.goals.vector_goals.lvlh_tangential_goal.LVLH_Tangential_Goal` (along-track),
:class:`~ADCS.CONOPS.goals.vector_goals.velocity_goal.Velocity_Goal`, and
:class:`~ADCS.CONOPS.goals.vector_goals.antivelocity_goal.AntiVelocity_Goal`.
The package also provides
:class:`~ADCS.CONOPS.goals.vector_goals.sun_goal.Sun_Goal`,
:class:`~ADCS.CONOPS.goals.vector_goals.antisun_goal.AntiSun_Goal`,
:class:`~ADCS.CONOPS.goals.vector_goals.bfield_goal.BField_Goal`,
:class:`~ADCS.CONOPS.goals.vector_goals.antibfield_goal.AntiBField_Goal`, and
:class:`~ADCS.CONOPS.goals.vector_goals.perpbfield_goal.PerpBField_Goal`
for directions set by the environment.
