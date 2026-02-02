# 🚀 Generalized ADCS

<p>
  <img src="docs/source/_static/starlab_logo.svg" alt="STARLab Logo" height="100">
  <img src="docs/source/_static/ssc_logo.png" alt="Small Satellite Collaborative Logo" height="100">
</p>

<strong>Generalized ADCS</strong> is a Python framework for satellite attitude determination
and control (ADCS), designed for **research, prototyping, and flight-software development**.
The framework emphasizes generality, modularity, and transparency for modern spacecraft
control and estimation workflows.

<p align="center">
  <a href="https://nscheuer.github.io/Generalized_ADCS/index.html">🛠 Documentation</a> •
  <a href="https://nscheuer.github.io/Generalized_ADCS/installation/index.html">📘 Installation</a> •
  <a href="https://nscheuer.github.io/Generalized_ADCS/tutorials/index.html">🧪 Tutorials</a> •
  <a href="https://nscheuer.github.io/Generalized_ADCS/contributing/index.html">🤝 Contributing</a>
</p>


## ✨ Key Features

<p align="center">
  <img src="docs/source/_static/ground_tracking.png"
       alt="Tracking a ground target"
       height="300">
</p>

- ✅ Fully generalized 6-DOF spacecraft attitude dynamics (RK4 integration)
- ✅ Fully generalized orbit propagation
- ✅ Estimation frameworks:
  UKF, SRUKF, UAKF, SRUAKF, orbital estimators, and custom filters
- ✅ Controller frameworks:
  PD, LQR, ALTRO, and user-defined controllers
- ✅ Sensor modeling:
  magnetometers, gyroscopes, sun sensors, GPS
- ✅ Actuator modeling:
  reaction wheels and magnetorquers
- ✅ Growing catalog of CubeSat-scale sensors and actuators
- ✅ Designed for underactuated and overactuated systems

## 📚 Academic Background

This project is based on the PhD research of **Patrick McKeen**:

- 🔗 Source Code:  
  https://github.com/patrickmckeen/PhD_Dissertation_Code
- 📄 Dissertation:  
  *Computational Methods to Improve Satellite Attitude Determination and Control
  with a Focus on Autonomy, Generalizability, and Underactuation*  
  https://dspace.mit.edu/handle/1721.1/158874

