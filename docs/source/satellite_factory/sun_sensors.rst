Sun Sensors
===========

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
     - BRITE dedicated Sun sensors
     - 
     - 
     - `GNB report <https://www.researchgate.net/publication/349758612_ADCS_Preliminary_Design_For_GNB>`__
     - BRITE-Austria
     - 6 faces; accuracy known.
   * - 🔵
     - Clyde Space 3U solar-array SunPair proxy
     - ``create_Clydespace_3U_array``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_sunpair>`
     - 
     - BeaverCube 1, BeaverCube 2
     - Axis + efficiency.
   * - 🟡
     - Hamamatsu S3931 Sun sensors
     - ``create_hamamatsu_s3931_sun_sensors``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_sunpair>`
     - `ASCE Library <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504>`__
     - ESTCube-1
     - ESTCube bias/noise defaults.
   * - 🟣
     - Solar MEMS NANO-ISS60 Sun sensors
     - ``create_nano_iss60_sun_sensors``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_sunpair>`
     - `mediaTUM <https://mediatum.ub.tum.de/doc/1483411/document.pdf>`__
     - MOVE-II
     - MOVE-II noise default.
   * - 🟡
     - OSRAM SFH2430 Sun sensors
     - ``create_osram_sfh2430_sun_sensors``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_sunpair>`
     - `Deep Blue <https://deepblue.lib.umich.edu/bitstream/handle/2027.42/140645/1.g000175.pdf?sequence=1>`__
     - RAX-1, RAX-2
     - RAX layout + noise default.
