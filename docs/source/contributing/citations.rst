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

Satellite citations should be added here when a spacecraft configuration uses
properties from a flown CubeSat, mission paper, thesis, technical report, or
public mission documentation.

Actuators & Sensors
-------------------

Actuator and sensor citations should be added here when default properties,
noise models, bias models, saturation limits, or geometry assumptions are taken
from papers, data sheets, or public hardware documentation.
