Citations
=========

This page tracks the source literature behind implemented algorithms and
documented hardware configurations. Each entry links to the relevant package
documentation, gives the original citation, and links to the original paper.

When adding a new paper-derived model, include the documentation page where the
model is described and cite the source used for the equations, assumptions, or
parameter values.

Control Laws
------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 40 15

   * - Model
     - Documentation
     - Original citation
     - Paper
   * - Lovera-Astolfi magnetic PD control
     - :doc:`MTQ_Lovera <../ADCS.controller.mtq_lovera>`
     - M. Lovera and A. Astolfi, "Global Magnetic Attitude Control of Inertially Pointing Spacecraft," *Journal of Guidance, Control, and Dynamics*, Vol. 28, No. 5, 2005, pp. 1065-1072.
     - `doi:10.2514/1.11844 <https://doi.org/10.2514/1.11844>`__
   * - Wisniewski sliding mode magnetic control
     - :doc:`MTQ_Wisniewski <../ADCS.controller.mtq_wisniewski>`
     - R. Wisniewski, "Sliding Mode Attitude Control for Magnetic Actuated Satellite," *IFAC Proceedings Volumes*, Vol. 31, No. 21, 1998, pp. 179-184.
     - `doi:10.1016/S1474-6670(17)41076-7 <https://doi.org/10.1016/S1474-6670(17)41076-7>`__
   * - Hogan-Schaub continuous momentum dumping
     - :doc:`MTQ_w_RW <../ADCS.controller.mtq_w_rw>`
     - E. A. Hogan and H. Schaub, "Three-Axis Attitude Control Using Redundant Reaction Wheels with Continuous Momentum Dumping," *Journal of Guidance, Control, and Dynamics*, Vol. 38, No. 10, 2015, pp. 1865-1871.
     - `doi:10.2514/1.G000812 <https://doi.org/10.2514/1.G000812>`__
   * - Mixed RW-MTQ LP allocation
     - :doc:`MTQ_w_RW_LP <../ADCS.controller.mtq_w_rw_LP>`
     - P. McKeen, N. Scheuer, and K. Cahoy, "Generalized Attitude Control for Small Spacecraft," 40th Annual Small Satellite Conference, Poster Session 2, SSC26-P2-54, 2026.
     - `USU DigitalCommons <https://digitalcommons.usu.edu/smallsat/2026/all2026/253/>`__
   * - Mixed RW-MTQ QP allocation
     - :doc:`MTQ_w_RW_QP <../ADCS.controller.mtq_w_rw_QP>`
     - P. McKeen, N. Scheuer, and K. Cahoy, "Generalized Attitude Control for Small Spacecraft," 40th Annual Small Satellite Conference, Poster Session 2, SSC26-P2-54, 2026.
     - `USU DigitalCommons <https://digitalcommons.usu.edu/smallsat/2026/all2026/253/>`__

Estimators
----------

Estimator citations should be added here when an estimator implementation is
derived from a specific paper or validated against a published formulation.

Satellites
----------

.. list-table::
   :header-rows: 1
   :widths: 20 25 40 15

   * - Model
     - Documentation
     - Original citation
     - Paper
   * - ESTCube-1
     - :doc:`CubeSat factories <../ADCS.satellite_factory.satellites.create_cubesats>`
     - J. Slavinskis et al., "ESTCube-1 attitude determination system flight results," *Journal of Aerospace Engineering*, 2016.
     - `ASCE Library <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504>`__
   * - RAX-1
     - :doc:`CubeSat factories <../ADCS.satellite_factory.satellites.create_cubesats>`
     - J. C. Springmann et al., "The attitude determination system of the RAX satellite," *Acta Astronautica*, Vol. 75, 2012, pp. 120-135.
     - `doi:10.1016/j.actaastro.2012.02.001 <https://doi.org/10.1016/j.actaastro.2012.02.001>`__
   * - RAX-2
     - :doc:`CubeSat factories <../ADCS.satellite_factory.satellites.create_cubesats>`
     - J. C. Springmann and J. W. Cutler, "Flight results of a low-cost attitude determination system," *Acta Astronautica*, Vol. 99, 2014, pp. 201-214.
     - `doi:10.1016/j.actaastro.2014.02.026 <https://doi.org/10.1016/j.actaastro.2014.02.026>`__

Actuators & Sensors
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 40 15

   * - Model
     - Documentation
     - Original citation
     - Paper
   * - ESTCube-1 ITG-3200 gyros, HMC5883L magnetometers, and Hamamatsu S3931 Sun sensors
     - :doc:`Sensor factories <../ADCS.satellite_factory.sensors>`
     - J. Slavinskis et al., "ESTCube-1 attitude determination system flight results," *Journal of Aerospace Engineering*, 2016.
     - `ASCE Library <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504>`__
   * - ESTCube-1 magnetic coils
     - :doc:`Actuator factories <../ADCS.satellite_factory.actuators>`
     - ESTCube-1 magnetic actuator flight-results paper, *Acta Astronautica*, 2016.
     - `ScienceDirect <https://www.sciencedirect.com/science/article/pii/S0094576515302216>`__
   * - RAX ADIS16405 gyros and magnetometers, MicroMag3 magnetometer, and SFH2430 Sun sensors
     - :doc:`Sensor factories <../ADCS.satellite_factory.sensors>`
     - J. C. Springmann, "Satellite Attitude Determination with Low-Cost Sensors," Ph.D. dissertation, University of Michigan, 2013.
     - `Deep Blue <https://deepblue.lib.umich.edu/bitstream/handle/2027.42/102312/jspringm_1.pdf?sequence=1>`__
   * - RAX OSRAM SFH2430 photodiode layouts and calibration
     - :doc:`Sensor factories <../ADCS.satellite_factory.sensors>`
     - J. C. Springmann and J. W. Cutler, "On-orbit calibration of photodiodes for attitude determination," *Journal of Guidance, Control, and Dynamics*, 2014.
     - `Deep Blue <https://deepblue.lib.umich.edu/bitstream/handle/2027.42/140645/1.g000175.pdf?sequence=1>`__
