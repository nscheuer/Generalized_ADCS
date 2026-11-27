# Attitude estimation

`AttitudeMEKF` owns a nominal scalar-first Hamilton quaternion (inertial to body),
body rates, and a 6×6 covariance in `[body attitude error, body rate error]`
coordinates. `AttitudeEKF` instead uses the seven-component state
`[q0, q1, q2, q3, wx, wy, wz]` and a 7×7 covariance. Its quaternion covariance
is projected onto the unit-quaternion tangent space after prediction and update.
Neither estimator includes a bias state. `AttitudeEstimator` validates initial
arrays; both estimators run prediction and sensor correction in Numba kernels
selected by `estimator/runtime.py` inside the simulation loop.

Attach either estimator with `ADCSSubsystem(estimator=AttitudeMEKF(q0, w0, P0, acceleration_psd), sensors=[...])`
or `ADCSSubsystem(estimator=AttitudeEKF(q0, w0, P0, acceleration_psd), sensors=[...])`.
The estimator runs at each subsystem tick `N * dt`, starting with the initial
readings. Enable `estimated_attitude`, `estimated_rate`, and
`estimated_covariance` in `ResultsFormat` to record its output. Covariance is
flattened row-major into 36 columns for the MEKF and 49 columns for the EKF;
samples between subsystem ticks are NaN.

The estimators can use mounted gyroscopes for rate propagation and correct
attitude from available autonomous star tracker, magnetometer, coarse/fine Sun,
and fine Earth sensor measurements. Magnetometer corrections compare the
measured body-frame field with the inertial magnetic field from the configured
environment model. A magnetometer therefore requires both an enabled sensor
and `environment.magnetic_field`; merely configuring a field model does not
enable magnetic corrections. Optical direction measurements likewise use
inertial references from the environment. Disabled or unavailable readings are
skipped. Known gyro bias is subtracted; hardware bias is never estimated.
Propagation assumes constant body rate with white angular acceleration. The
initial attitude and rate must be supplied by the caller; there is no
measurement-based initializer.
