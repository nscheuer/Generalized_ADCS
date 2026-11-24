# Attitude estimation

`AttitudeMEKF` owns a nominal scalar-first Hamilton quaternion (inertial to body),
body rates, and a 6×6 covariance in `[body attitude error, body rate error]`
coordinates. There is no bias state. `AttitudeEstimator` validates and owns
initial arrays; `attitude_mekf.py` implements Numba prediction and correction kernels;
`estimator/runtime.py` selects the compiled estimator inside the simulation
loop. New estimators can inherit the input contract and add a numeric dispatch
case without sending sensor ticks back through Python.

Attach with `ADCSSubsystem(estimator=AttitudeMEKF(q0, w0, P0, acceleration_psd),
sensors=[...])`. The estimator runs at each subsystem tick `N * dt`, starting
with the initial readings. Enable `estimated_attitude`, `estimated_rate`, and
`estimated_covariance` in `ResultsFormat` to record its output. Covariance is
flattened row-major into 36 columns; samples between subsystem ticks are NaN.

The current measurement set is mounted gyroscopes and autonomous star trackers.
Disabled or unavailable readings are skipped. Known gyro bias is subtracted;
hardware bias is never estimated. Propagation assumes constant body rate with
white angular acceleration. The initial attitude and rate must be supplied by
the caller; there is no measurement-based initializer.
