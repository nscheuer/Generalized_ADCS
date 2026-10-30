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

Aerodynamics
------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 40 15

   * - Model
     - Documentation
     - Original citation
     - Paper or source
   * - Panel drag and ballistic coefficient
     - :doc:`Drag_Disturbance <../ADCS.satellite_hardware.disturbances.drag_disturbance>`
     - The implementation uses the standard quadratic aerodynamic drag law, resolved over body-fixed faces. See D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Microcosm Press (2013), atmospheric drag discussion.
     - `CelesTrak TLE/B* FAQ <https://celestrak.org/columns/v04n03/>`__; `Vallado et al., AIAA-2006-6753 <https://celestrak.org/publications/AIAA/2006-6753/>`__
   * - Sentman free-molecular drag
     - :py:class:`~ADCS.satellite_hardware.disturbances.sentman_drag_disturbance.Sentman_Drag_Disturbance`
     - L. H. Sentman, *Free Molecule Flow Theory and Its Application to the Determination of Aerodynamic Forces*, Lockheed Missiles & Space Company, report LMSC-448514 (1961). The diffuse re-emission and gas-surface assumptions are also informed by K. Moe and M. M. Moe, "Gas–surface interactions and satellite drag coefficients," *Planetary and Space Science*, Vol. 53, No. 8, 2005, pp. 793–801, and E. Doornbos, *Thermospheric Density and Wind Determination from Satellite Dynamics*, Springer (2012), section 3.4.
     - `Sentman report <https://citeseerx.ist.psu.edu/document?doi=474aab2a5070e9c012e0d6f21b43d11c838207b3&repid=rep1&type=pdf>`__; `Moe & Moe, doi:10.1016/j.pss.2005.03.005 <https://doi.org/10.1016/j.pss.2005.03.005>`__; `Doornbos, doi:10.1007/978-3-642-25129-0 <https://doi.org/10.1007/978-3-642-25129-0>`__

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
   * - SGP4 propagation from TLEs
     - :doc:`SGP4Orbit <../ADCS.orbits.sgp4_orbit>`
     - D. A. Vallado, P. Crawford, R. Hujsak, and T. S. Kelso, “Revisiting Spacetrack Report #3,” AIAA/AAS Astrodynamics Specialist Conference, AIAA-2006-6753 (2006). The implementation uses the ``sgp4`` package and Skyfield's vectorized SGP4 interface.
     - `CelesTrak paper, source code, and test cases <https://celestrak.org/publications/AIAA/2006-6753/>`__; `python-sgp4 documentation <https://pypi.org/project/sgp4/>`__; `Skyfield documentation <https://rhodesmill.org/skyfield/earth-satellites.html>`__
   * - TLE format and SGP4 B* drag term
     - :doc:`SGP4Orbit <../ADCS.orbits.sgp4_orbit>`
     - CelesTrak documents B* as the SGP4 drag-like parameter, adjusted from the aerodynamic ballistic coefficient. The geometry-derived estimate in this package uses the conventional reference-density relation and an isotropic mean projected-area assumption.
     - `CelesTrak TLE format and B* definition <https://celestrak.org/columns/v04n03/>`__; `NASA HST/Swift orbit-decay study, NASA/TM-20250010561 (2025), reference-density convention <https://ntrs.nasa.gov/citations/20250010561>`__
   * - RK4 orbit propagation and force model
     - :doc:`RK4Orbit <../ADCS.orbits.rk4_orbit>`
     - Classical fourth-order Runge–Kutta integration is used for Cowell-form translational dynamics. The modeled perturbations include Earth zonal gravity, third-body gravity, and spacecraft aerodynamic force.
     - `Butcher, “Coefficients for the Study of Runge-Kutta Integration Processes,” doi:10.1093/comjnl/6.1.50 <https://doi.org/10.1093/comjnl/6.1.50>`__; Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed. (2013)
   * - Earth gravitational parameter, reference radius, and zonal coefficients
     - :doc:`EarthConstants <../ADCS.orbits.universal_constants>`
     - The package uses EGM96-derived unnormalized zonal coefficients through degree 6. The selected constants and coefficient convention are documented with ``EarthConstants``.
     - `Lemoine et al., NASA/TP-1998-206861, EGM96 <https://ntrs.nasa.gov/citations/19980218814>`__; `NGA WGS 84 reference constants <https://earth-info.nga.mil/?action=wgs84&dir=wgs84>`__
   * - Solar and lunar gravitational parameters
     - :doc:`RK4Orbit <../ADCS.orbits.rk4_orbit>`
     - Third-body accelerations use the Sun and Moon gravitational parameters.
     - `JPL Solar System Dynamics, Astrodynamic Parameters <https://ssd.jpl.nasa.gov/astro_par.html>`__
   * - Planetary ephemeris and celestial-body positions (default DE421)
     - ``ADCS.environment.ephemeris.DE421``; alternatives: ``DE442``, ``DE442s``, ``DE441``
     - W. M. Folkner, J. G. Williams, and D. H. Boggs, *The Planetary and Lunar Ephemeris DE421*, JPL IOM 343.R-08-003 (2008). Skyfield provides the Python interface used to read these JPL kernels.
     - `JPL DE421 references <https://iaaras.ru/en/dept/ephemeris/online/references/>`__; `Skyfield ASCL: 1907.024 <https://ascl.net/1907.024>`__
   * - IGRF geomagnetic field
     - ``ADCS.orbits.orbital_state.Orbital_State``
     - International Association of Geomagnetism and Aeronomy, *IGRF-14* (2024), coefficient dataset. Also cite C. Beggan et al., "International geomagnetic reference field: the fourteenth generation," *Earth, Planets and Space*, 78, 127 (2026), and IAGA-VMOD, *ppigrf: Pure Python IGRF* for the legacy backend.
     - `IGRF-14 coefficient dataset, doi:10.5281/zenodo.14012302 <https://doi.org/10.5281/zenodo.14012302>`__; `IGRF-14 paper, doi:10.1186/s40623-025-02360-0 <https://doi.org/10.1186/s40623-025-02360-0>`__; `ppigrf <https://github.com/IAGA-VMOD/ppigrf>`__
   * - WMM2025 geomagnetic field
     - ``ADCS.environment.magnetic_field.WMM2025``
     - U.S. National Oceanic and Atmospheric Administration (NOAA), National Centers for Environmental Information (NCEI), *World Magnetic Model 2025*; developed with the British Geological Survey (BGS).
     - `NOAA WMM 2025 <https://www.ncei.noaa.gov/products/world-magnetic-model>`__
   * - CHAOS-8.7 internal geomagnetic field
     - ``ADCS.environment.magnetic_field.CHAOS8``
     - DTU Space geomagnetic field model team, CHAOS-8.7 core and lithospheric coefficient data.
     - `CHAOS-8.7 data, Zenodo record 22959079 <https://zenodo.org/records/22959079>`__
   * - NRLMSISE-00 neutral atmosphere
     - ``ADCS.environment.atmosphere.NRLMSISE00``
     - J. M. Picone, A. E. Hedin, D. P. Drob, and A. C. Aikin, “NRLMSISE-00 empirical model of the atmosphere: Statistical comparisons and scientific issues,” *J. Geophys. Res.*, 107(A12), 1468 (2002), doi:10.1029/2002JA009430. Developed at the U.S. Naval Research Laboratory (NRL).
     - `doi:10.1029/2002JA009430 <https://doi.org/10.1029/2002JA009430>`__
   * - Standard tabulated atmosphere (SMAD)
     - ``ADCS.environment.atmosphere.SMADAtmosphere``
     - J. R. Wertz and W. J. Larson (eds.), *Space Mission Analysis and Design*, 3rd ed., Microcosm Press and Kluwer Academic Publishers (1999). This implementation uses tabulated reference densities and log-linear interpolation.
     - `Publisher information <https://www.microcosm.com/>`__
   * - Bright Star Catalogue
     - ``ADCS.environment.star_catalogue.BrightStarCatalog``
     - D. Hoffleit and W. H. Warren, Jr., *The Bright Star Catalogue*, 5th Revised Edition, Yale University Observatory (1991); distributed by CDS Strasbourg as catalogue V/50.
     - `CDS V/50 ReadMe <https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/V/50?format=html&tex=true>`__
   * - Hipparcos main catalogue
     - ``ADCS.environment.star_catalogue.HipparcosCatalog``
     - ESA, *The Hipparcos and Tycho Catalogues*, ESA SP-1200 (1997); catalogue I/239 distributed by CDS Strasbourg.
     - `CDS I/239 ReadMe <https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/I/239?format=html&tex=true>`__

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
