Features (2026-01-28)
=====================

🚀 **Dynamics & Environment**
------------------------------
- Fully generalized 6-DOF attitude and orbit dynamics (RK4 integration)
- Built-in disturbance models:
  Gravity Gradient, Atmospheric Drag, SRP, Magnetic Dipole, Propulsion

🛰️ **Actuators & Sensors**
---------------------------
- Reaction wheels and magnetorquers
- Gyroscopes, magnetometers, sun sensors, star trackers, and GPS
- Noise and random-walk bias models for all sensors and actuators

🧠 **Control & Estimation**
----------------------------
- Plug-and-play controllers supporting underactuated and overactuated systems
- Estimators with augmented states for attitude, bias, and disturbance estimation

🎯 **Guidance & Planning**
---------------------------
- Support for multi-goal and time-varying trajectories
- Optional trajectory optimization framework

🧩 **Simulation & Execution**
------------------------------
- Basic scheduling interface for single-core onboard processors
- Fast saving and loading of orbits and full simulation results

📦 **Models & Reusability**
----------------------------
- Growing catalog of pre-defined satellites, sensors, and actuators
