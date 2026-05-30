0.1.5 Bugfixes and Optimizations (2026-05-29)
=====================================================================

Over the last few weeks, we have worked on a number of quality-of-life improvements, bug fixes, and optimizations across the codebase. Tests were added to bugfixes to ensure regression does not happen.

Bugs
----
- Fixed a bug where the reaction wheel torque() and storage_torque() methods would independently sample noise and bias values, leading to a violation of Newton's 3rd law
- Reaction wheel momentum saturation was sign-blind and only checked saturation in the positive direction
- Fixed dimensions and bugs for the disturbance jacabians, across GG, SRP, Drag, and Dipole disturbance models
- Fixed a bug in the ecef_to_geocentric() conversion
- Fixed a bug in the orbital_state timing, using TT instead of TAI time for certain operations
- Fixed a bug in the extrapolation of the atmospheric density model
- Fixed a bug in the sun sensor class, where noise and bias was lagged by one simulation step
- Fixed an issue where imported repositories (SALTRO and OldPlanner) would have pybind name resolution clashes
- Fixed a bug where the simulate() function would not handle satellites with bias and estimators without active bias estimation
- Fixed a bug where estimators would not properly deepcopy disturbance, actuator, and sensor models for noise independence from the real satellite
- Fixed disturbance-parameter estimation, which was not running properly despite being configured
- Fixed a bug where the orbit GPS estimator would not build the covariance matrix properly, leading to NaN values and divergence

Changes
-------
- Made actuator saturation clamping strict instead of just outputting a warning
- Made orbital averaging use RK4 instead of midpoint integration for higher accuracy with no significant performance penalty
- Tuned the baseline SRUKF estimator parameters for better convergence and stability
- Added documentation for MacOS use

Test Coverage
-------------
- Added test coverage for reaction wheel momentum saturation and clamping
- Added test coverage for disturbance jacobians
- Added test coverage for second derivative helper functions
- Added test coverage for njit kernel outputs and shapes
- Added test coverage for estimator augmented state convergence
- Added test coverage for the GPS EKF estimator
- Added test coverage for the GoalList class
- Added test coverage for the Monte Carlo simulation framework