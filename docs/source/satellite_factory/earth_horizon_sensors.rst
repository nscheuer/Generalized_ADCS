Earth Horizon Sensors
=====================

Tier Legend
-----------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Tier
     - Meaning
   * - ⚪
     - Representative or generic Earth-horizon sensor layout/defaults.
   * - 🔵
     - Geometry and measurement layout are specified; error defaults may be representative or assumed.
   * - 🟣
     - Blue-tier information plus source-backed or explicit measurement-noise defaults.
   * - 🟡
     - Purple-tier information plus source-backed bias or calibration defaults.

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
   * - ⚪
     - Generic Earth horizon sensor
     - ``create_generic_earth_horizon``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_earth_horizon>`
     - 
     - -
     - Configurable FOV + noise.
   * - ⚪
     - IRST Earth horizon sensor
     - ``create_irst_horizon_sensor``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_earth_horizon>`
     - 
     - -
     - IRST FOV + noise preset.
