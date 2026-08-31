Gyroscopes
==========

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
     - ICM-20948 IMU gyros
     - ``create_ICM20948_IMU``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_gyro>`
     - 
     - BeaverCube 1, BeaverCube 2
     - 3-axis layout.
   * - 🟡
     - ADIS16405 gyros
     - ``create_adis16405_gyros``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_gyro>`
     - `Deep Blue <https://deepblue.lib.umich.edu/bitstream/handle/2027.42/102312/jspringm_1.pdf?sequence=1>`__
     - RAX-1, RAX-2
     - RAX bias/noise defaults.
   * - 🟡
     - Bosch BMX055 gyros
     - ``create_bmx055_gyros``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_gyro>`
     - `mediaTUM <https://mediatum.ub.tum.de/doc/1483411/document.pdf>`__
     - MOVE-II
     - MOVE-II bias/noise defaults.
   * - 🟡
     - InvenSense ITG-3200 gyros
     - ``create_itg3200_gyros``
     - :doc:`Factory <../ADCS.satellite_factory.sensors.create_cubesat_gyro>`
     - `ASCE Library <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504>`__
     - ESTCube-1
     - ESTCube bias/noise defaults.
