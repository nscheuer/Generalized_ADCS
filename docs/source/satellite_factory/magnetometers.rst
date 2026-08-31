Magnetometers
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
     - ISIS magnetometer triad
     - ``create_isis_magnetometer``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_MTM>`
     - 
     - BeaverCube 1, BeaverCube 2
     - 3-axis layout.
   * - 🟡
     - ADIS16405 magnetometers
     - ``create_adis16405_magnetometers``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_MTM>`
     - `Deep Blue <https://deepblue.lib.umich.edu/bitstream/handle/2027.42/102312/jspringm_1.pdf?sequence=1>`__
     - RAX-1, RAX-2
     - RAX bias/noise defaults.
   * - 🟡
     - Bosch BMX055 magnetometers
     - ``create_bmx055_magnetometers``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_MTM>`
     - `mediaTUM <https://mediatum.ub.tum.de/doc/1483411/document.pdf>`__
     - MOVE-II
     - MOVE-II bias/noise defaults.
   * - 🟡
     - Honeywell HMC5883L magnetometers
     - ``create_hmc5883l_magnetometers``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_MTM>`
     - `ASCE Library <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504>`__
     - ESTCube-1
     - ESTCube bias/noise defaults.
   * - 🟡
     - MicroMag3 magnetometers
     - ``create_micromag3_magnetometers``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_MTM>`
     - `Deep Blue <https://deepblue.lib.umich.edu/bitstream/handle/2027.42/102312/jspringm_1.pdf?sequence=1>`__
     - RAX-1, RAX-2
     - RAX noise default.
