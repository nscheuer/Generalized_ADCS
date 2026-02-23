0.0.0 Initial Release (2026-01-15)
=============================================

Initial Features
----------------
- **Actuators**: Magnetorquers (MTQ) and Reaction Wheels (RW)
- **Satellites**: Configurable satellite models with mass, inertia, actuators, and sensors
- **Controllers**:
    - MTQ_w_RW (Hogan and Schaub)
    - Wisniewski
    - Lovera
- **Control Allocations**: Linear Programming (LP), Quadratic Programming (QP), Weighted QP
- **Noise and Bias**: Random-walk bias models for sensors and actuators
- **Sensors**: Gyroscope, Sun Sensor, GPS, Magnetometer, Star Tracker
- **Orbit Calculation and Propagation**: J2 perturbation modeling and orbit state propagation
- **Estimators**: Square-Root Unscented Additive Kalman Filter (SRUAKF), Unscented Additive Kalman Filter (UAKF)
- **simulate() Framework**: Easy-to-use simulation framework with integrated plotting utilities

For tutorials and full feature demonstrations, see :doc:`../tutorials/index`.

