Star Trackers
=============

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
   * - 🔵
     - AeroAstro MST
     - 
     - 
     - `TU Graz <https://www.tugraz.at/en/institute/iks/space-missions/brite-austria-tugsat-1/satellite-and-payload>`__
     - BRITE-Austria
     - Telescope-aligned attitude sensor.
   * - 🟣
     - BCT Nano Star Tracker
     - ``create_bct_nst``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_star_tracker>`
     - 
     - -
     - Boresight, FOV, exclusion, noise.
   * - 🟣
     - Terma T1 Star Tracker
     - ``create_terma_t1``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_star_tracker>`
     - 
     - -
     - Boresight, FOV, exclusion, noise.
   * - 🟣
     - Generic vector star tracker
     - ``create_generic_star_tracker``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_star_tracker>`
     - 
     - -
     - Configurable vector tracker.
   * - 🟣
     - BCT Nano Star Tracker, quaternion output
     - ``create_bct_nst_quaternion``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_star_tracker>`
     - 
     - -
     - Quaternion output preset.
   * - 🟣
     - Generic quaternion star tracker
     - ``create_generic_star_tracker_quaternion``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_star_tracker>`
     - 
     - -
     - Configurable quaternion tracker.
