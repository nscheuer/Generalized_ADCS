Satellite Factory
=================

These pages summarize the preset spacecraft and hardware factories available in
``ADCS.satellite_factory``.

For component-only factory pages, the tier is applied to the component layout
and error-model information available for that hardware. Satellite inertia and
center of mass are evaluated on the satellite rows.

Tier Legend
-----------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Tier
     - Meaning
   * - ⚪
     - Estimates on satellite, actuator, and sensor layout.
   * - 🔵
     - Satellite, actuator, and sensor layout known.
   * - 🟣
     - Blue-tier information plus noise known.
   * - 🟡
     - Purple-tier information plus noise and bias known.

.. toctree::
   :maxdepth: 1

   satellites
   reaction_wheels
   magnetorquers
   magnetometers
   gyroscopes
   sun_sensors
   star_trackers
   gps
   earth_horizon_sensors
