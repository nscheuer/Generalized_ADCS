# 🚀 Generalized ADCS
<p>
  <img src="documentation/images/starlab_logo.svg" alt="STARLab Logo" height="100">
  <img src="documentation/images/ssc_logo.png" alt="Small Satellite Collaborative Logo" height="100">
</p>
<strong>Generalized ADCS</strong> is a Python framework for satellite attitude determination and control (ADCS), designed for research, prototyping, and flight-software development.

<p align="center"> <a href="/documentation/installation_instructions/INSTALL.md">📘 Installation</a> • <a href="/examples">🧪 Examples</a> • <a href="https://nscheuer.github.io/Generalized_ADCS/">🛠 Documentation</a> • <a href="/CONTRIBUTING.md">🤝 Contributing</a> </p>

## ✨ Key Features 
<p align="center">
  <img src="documentation/images/ground_tracking.png" alt="Tracking a ground target" height="300">
</p>

- ✅ Fully generalized 6-DOF RK4 spacecraft dynamics
- ✅ Fully generalized RK4 orbit propagation
- ✅ Estimator support (Orbital, UKF, SRUKF, UAKF, SRUAKF, custom filters)
- ✅ Controller support (PD, LQR, Trajectory Planning, custom controllers)
- ✅ Sensor Modeling
    - Magnetometers
    - Gyroscopes
    - Sun Sensors
    - GPS
- ✅ Actuator Modeling
    - Reaction Wheels
    - Magnetorquers
- ✅ Growing catalogue of CubeSat sensors and actuators

## 📚 Academic Background
This project is based on the PhD research of <strong>Patrick McKeen</strong>:
- 🔗 Source Code: 
[PhD Dissertation Code](https://github.com/patrickmckeen/PhD_Dissertation_Code)
- 📄 Thesis: [*Computational Methods to Improve Satellite Attitude Determination and Control with a Focus on Autonomy, Generalizability, and Underactuation*](https://dspace.mit.edu/handle/1721.1/158874)

Simulation suite for satellite control and estimation. Based on work by Patrick McKeen [GitHub](https://github.com/patrickmckeen/PhD_Dissertation_Code "PhD_Dissertation_Code"). [Paper](https://dspace.mit.edu/handle/1721.1/158874 "Computational Methods to Improve Satellite Attitude Determination and Control with a Focus on Autonomy, Generalizability, and Underactuation").

## ⚡ Quick Start
```bash
git clone https://github.com/nscheuer/Generalized_ADCS.git
cd Generalized_ADCS
pip install -r requirements.txt
pip install git+https://github.com/jcrudy/choldate.git --no-build-isolation
python examples/cubesat_examples/beavercube1_base_estimator_noisy.py
```
For full installation instructions for your system, including compiling the <u>trajectory planner</u>, see [📘 Installation](/documentation/installation_instructions/INSTALL.md).

# Compiling Documentation
Instructions on how to compile the Sphinx documentation [here](/documentation/SPHINX.md).