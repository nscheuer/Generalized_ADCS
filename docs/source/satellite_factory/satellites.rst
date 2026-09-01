Satellites
==========

.. list-table::
   :class: sortable-factory
   :header-rows: 1
   :widths: 7 13 19 12 12 12 19 15

   * - Tier
     - Model
     - Factory
     - Documentation
     - Source
     - Timeline
     - Goal
     - Configuration
   * - 🔵
     - BeaverCube 1
     - ``create_beavercube1_cubesat``
     - :doc:`Factory <../ADCS.satellite_factory.satellites.create_cubesats>`
     - 
     - Launched 2022-09-06; mission ended 2023-04-12.
     - Coastal imaging mission using visible and thermal infrared CubeSat data.
     - 3x MTQ, 3x MTM, 3x gyro, 2x SunPair.
   * - 🔵
     - BeaverCube 2
     - ``create_beavercube2_cubesat``
     - :doc:`Factory <../ADCS.satellite_factory.satellites.create_cubesats>`
     - 
     - Slated March 2027.
     - Demonstrate autonomous on-orbit image processing and classification.
     - 3x MTQ, 1x RW, 3x MTM, 3x gyro, 2x SunPair.
   * - 🔵
     - BeaverCube 2, 3 MTQ + 3 RW
     - ``create_3_3_beavercube2_cubesat``
     - :doc:`Factory <../ADCS.satellite_factory.satellites.create_cubesats>`
     - 
     - Theoretical study.
     - BeaverCube 2 variant for full three-wheel control studies.
     - 3x MTQ, 3x RW, 3x MTM, 3x gyro, 2x SunPair.
   * - ⚪
     - BRITE-Austria
     - 
     - 
     - `TU Graz <https://www.tugraz.at/en/institute/iks/space-missions/brite-austria-tugsat-1/satellite-and-payload>`__
     - Launched 2013-02-25.
     - High-precision photometry of bright massive stars.
     - 3x RW, 3x MTQ, 3x gyro, 1x MTM, 6x SunSensor, 1x StarTracker.
   * - 🟡
     - ESTCube-1
     - ``create_estcube1_cubesat``
     - :doc:`Factory <../ADCS.satellite_factory.satellites.create_cubesats>`
     - `ASCE Library <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504>`__
     - Launched 2013-05-07; mission ended 2015-02-17.
     - Demonstrate electric solar wind sail technology from a student CubeSat.
     - 3x MTQ, 6x MTM, 12x gyro, 12x SunSensor.
   * - 🟡
     - MOVE-II
     - ``create_moveii_cubesat``
     - :doc:`Factory <../ADCS.satellite_factory.satellites.create_cubesats>`
     - `doi:10.3390/aerospace6120130 <https://doi.org/10.3390/aerospace6120130>`__
     - Launched 2018-12-03.
     - Verify a CubeSat bus able to support demanding payloads.
     - 3x MTQ, 18x MTM, 18x gyro, 5x SunSensor.
   * - 🟡
     - RAX-1
     - ``create_rax1_cubesat``
     - :doc:`Factory <../ADCS.satellite_factory.satellites.create_cubesats>`
     - `doi:10.1016/j.actaastro.2012.02.001 <https://doi.org/10.1016/j.actaastro.2012.02.001>`__
     - Launched 2010-11-19; ended after about 60 days.
     - Study small-scale plasma density irregularities in Earth's ionosphere.
     - Passive magnetic stabilization, 6x MTM, 3x gyro, 9x SunSensor.
   * - 🟡
     - RAX-2
     - ``create_rax2_cubesat``
     - :doc:`Factory <../ADCS.satellite_factory.satellites.create_cubesats>`
     - `doi:10.1016/j.actaastro.2014.02.026 <https://doi.org/10.1016/j.actaastro.2014.02.026>`__
     - Launched 2011-10-28; completed mission 2013-04.
     - Continue RAX ionospheric FAI science with improved bus performance.
     - Passive magnetic stabilization, 6x MTM, 3x gyro, 17x SunSensor.
