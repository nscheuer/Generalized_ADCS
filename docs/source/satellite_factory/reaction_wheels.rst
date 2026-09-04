Reaction Wheels
===============

Tier Legend
-----------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Tier
     - Meaning
   * - ⚪
     - Representative or theoretical wheel layout/defaults.
   * - 🔵
     - Wheel axis/layout and authority limits are traceable or explicitly stated.
   * - 🟣
     - Blue-tier information plus source-backed or clearly inferred wheel inertia or momentum data.
   * - 🟡
     - Purple-tier information plus source-backed actuator noise and bias/error model defaults.

.. list-table::
   :class: sortable-factory
   :header-rows: 1
   :widths: 8 18 22 14 14 14 10

   * - Tier
     - Model
     - Factory
     - Documentation
     - Source
     - Missions
     - Notes
   * - 🟣
     - BRITE/SFL reaction wheel
     - ``create_sfl_reaction_wheels``
     - :doc:`Factory <../ADCS.satellite_factory.actuators.create_cubesat_RW>`
     - `GNB report <https://www.researchgate.net/publication/349758612_ADCS_Preliminary_Design_For_GNB>`__
     - BRITE-Austria
     - Preliminary design values; not flight-calibrated.
   * - ⚪
     - CubeWheel Small+
     - ``create_cubewheel_smallplus_rw``
     - :doc:`Factory <../ADCS.satellite_factory.actuators.create_cubesat_RW>`
     - 
     - BeaverCube 2
     - Layout + wheel defaults.
   * - 🟣
     - LightSail 2 momentum wheel
     - ``create_sinclair_interplanetary_momentum_wheel``
     - :doc:`Factory <../ADCS.satellite_factory.actuators.create_cubesat_RW>`
     - `doi:10.3390/aerospace10070579 <https://www.mdpi.com/2226-4310/10/7/579>`__
     - LightSail 2
     - +Y wheel; inertia inferred from momentum/speed.
