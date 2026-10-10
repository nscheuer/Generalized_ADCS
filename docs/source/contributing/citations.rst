Citations
=========

This page tracks source literature behind package-level algorithms, estimator
formulations, environmental models, and reusable technical methods. Each entry
links to the relevant package documentation, gives the original citation, and
links to the original paper.

Satellite and component data sources for preset factories are listed on the
Satellite Factory pages next to the affected spacecraft or hardware rows. Keep
factory-only data sources there so large preset tables remain traceable without
turning this page into a duplicate bibliography.

When adding a new paper-derived algorithm or reusable model, include the
documentation page where the model is described and cite the source used for the
equations, assumptions, or parameter values.

Project information and the usage disclaimer are available on the
:doc:`About page <about>`.

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

Orbits
------

The orbit environment models use the following external Python libraries for
planetary ephemerides and geomagnetic-field calculations.

.. list-table::
   :header-rows: 1
   :widths: 25 25 35 15

   * - Model
     - Documentation
     - Original citation
     - Reference
   * - Planetary ephemeris and celestial-body positions
     - ``ADCS.orbits.ephemeris.Ephemeris``
     - B. Rhodes, "Skyfield: High precision research-grade positions for planets and Earth satellites in Python," 2019, ASCL: `ascl:1907.024 <https://ascl.net/1907.024>`__.
     - `Skyfield documentation <https://rhodesmill.org/skyfield/>`__
   * - IGRF geomagnetic field
     - ``ADCS.orbits.orbital_state.Orbital_State``
     - International Association of Geomagnetism and Aeronomy, *IGRF-14* (2024), coefficient dataset. Also cite C. Beggan et al., "International geomagnetic reference field: the fourteenth generation," *Earth, Planets and Space*, 78, 127 (2026), and IAGA-VMOD, *ppigrf: Pure Python IGRF* for the legacy backend.
     - `IGRF-14 coefficient dataset, doi:10.5281/zenodo.14012302 <https://doi.org/10.5281/zenodo.14012302>`__; `IGRF-14 paper, doi:10.1186/s40623-025-02360-0 <https://doi.org/10.1186/s40623-025-02360-0>`__; `ppigrf <https://github.com/IAGA-VMOD/ppigrf>`__

Estimators
----------

Estimator citations should be added here when an estimator implementation is
derived from a specific paper or validated against a published formulation.

.. list-table::
   :header-rows: 1
   :widths: 20 25 40 15

   * - Model
     - Documentation
     - Original citation
     - Paper
   * - Wahba's problem (the loss every single-frame method minimises)
     - :doc:`attitude_determination <../ADCS.estimators.attitude_determination>`
     - G. Wahba, "A Least Squares Estimate of Satellite Attitude," *SIAM Review*, Vol. 7, No. 3, 1965, p. 409.
     - `doi:10.1137/1007077 <https://doi.org/10.1137/1007077>`__
   * - TRIAD two-vector attitude determination
     - :doc:`attitude_determination <../ADCS.estimators.attitude_determination>`
     - H. D. Black, "A Passive System for Determining the Attitude of a Satellite," *AIAA Journal*, Vol. 2, No. 7, 1964, pp. 1350-1351.
     - `doi:10.2514/3.2555 <https://doi.org/10.2514/3.2555>`__
   * - Davenport's q-method
     - :doc:`attitude_determination <../ADCS.estimators.attitude_determination>`
     - P. B. Davenport, "A Vector Approach to the Algebra of Rotations with Applications," NASA TN D-4696, 1968.
     - `NTRS 19680021122 <https://ntrs.nasa.gov/citations/19680021122>`__
   * - QUEST and the optimal attitude covariance
     - :doc:`attitude_determination <../ADCS.estimators.attitude_determination>`
     - M. D. Shuster and S. D. Oh, "Three-Axis Attitude Determination from Vector Observations," *Journal of Guidance and Control*, Vol. 4, No. 1, 1981, pp. 70-77.
     - `doi:10.2514/3.19717 <https://doi.org/10.2514/3.19717>`__
   * - Quaternion from a rotation matrix (Shepperd's method)
     - :doc:`attitude_determination <../ADCS.estimators.attitude_determination>`
     - S. W. Shepperd, "Quaternion from Rotation Matrix," *Journal of Guidance and Control*, Vol. 1, No. 3, 1978, pp. 223-224.
     - `doi:10.2514/3.55767b <https://doi.org/10.2514/3.55767b>`__
   * - Survey of single-frame attitude estimators
     - :doc:`attitude_determination <../ADCS.estimators.attitude_determination>`
     - F. L. Markley and D. Mortari, "Quaternion Attitude Estimation Using Vector Observations," *Journal of the Astronautical Sciences*, Vol. 48, No. 2-3, 2000, pp. 359-380.
     - (journal article)
