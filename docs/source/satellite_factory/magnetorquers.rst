Magnetorquers
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
     - BRITE air-core coils
     - 
     - 
     - `USU DigitalCommons <https://digitalcommons.usu.edu/smallsat/2004/All2004/24/>`__
     - BRITE-Austria
     - Orthogonal coils; dipole known.
   * - 🔵
     - ISIS Magnetorquer Board
     - ``create_isis_magnetorquer_board``
     - :doc:`Factory <../ADCS.satellite_factory.actuators.create_cubesat_MTQ>`
     - 
     - BeaverCube 1, BeaverCube 2
     - 3-axis layout + defaults.
   * - 🔵
     - ESTCube-1 electromagnetic coils
     - ``create_estcube1_magnetorquers``
     - :doc:`Factory <../ADCS.satellite_factory.actuators.create_cubesat_MTQ>`
     - `ScienceDirect <https://www.sciencedirect.com/science/article/pii/S0094576515302216>`__
     - ESTCube-1
     - Body-axis coils; nominal dipole.
   * - 🔵
     - MOVE-II PCB magnetorquers
     - ``create_moveii_pcb_magnetorquers``
     - :doc:`Factory <../ADCS.satellite_factory.actuators.create_cubesat_MTQ>`
     - `EUCASS <https://www.eucass.eu/doi/EUCASS2017-664.pdf>`__
     - MOVE-II
     - Equivalent 3-axis PCB coils.
