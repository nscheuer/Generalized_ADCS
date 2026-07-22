# Generalized ADCS

<p>
  <img src="docs/source/_static/starlab_logo.svg" alt="STARLab Logo" height="100">
  <img src="docs/source/_static/ssc_logo.png" alt="Small Satellite Collaborative Logo" height="100">
</p>

<strong>Generalized ADCS</strong> is a Python framework for satellite attitude determination
and control (ADCS), designed for **research, prototyping, and flight-software development**.
The framework emphasizes generality, modularity, and transparency for modern spacecraft
control and estimation workflows.

<p align="center">
  <a href="https://nscheuer.github.io/Generalized_ADCS/index.html">Documentation</a> •
  <a href="https://nscheuer.github.io/Generalized_ADCS/installation/index.html">Installation</a> •
  <a href="https://nscheuer.github.io/Generalized_ADCS/tutorials/index.html">Tutorials</a> •
  <a href="https://nscheuer.github.io/Generalized_ADCS/contributing/index.html">Contributing</a>
</p>


## Key Features

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

Optional add-ons:
- trajectory_planner (tplaunch/pysat) and SALTRO (saltro_py) are optional C++ extensions.
- Core ADCS functionality works without them.
- Build instructions are in docs/Install_WSL.md and docs/Install_Windows.md.

## Academic Background

This project is based on the PhD research of **Patrick McKeen**:

- Source Code:  
  https://github.com/patrickmckeen/PhD_Dissertation_Code
- Dissertation:  
  *Computational Methods to Improve Satellite Attitude Determination and Control
  with a Focus on Autonomy, Generalizability, and Underactuation*  
  https://dspace.mit.edu/handle/1721.1/158874

See the release paper:

- *A Modular Open-Source ADCS Framework for Small Satellite Development and Testing*  
  https://smallsateurope.com/paper/a-modular-open-source-adcs-framework-for-small-satellite-development-and-testing/

## Used By

<p align="center"><em>Institutions using Generalized ADCS in research, coursework, and CubeSat development.</em></p>

<table border="0" cellspacing="0" cellpadding="20" align="center">
  <tr>
    <td align="center" width="260">
      <a href="https://tufts-cubesat.vercel.app/">
        <img src="https://brand.tufts.edu/sites/g/files/lrezom786/files/styles/large/public/2022-09/Tufts-logo-4c_5.jpg?itok=pQ14NYSX"
             alt="Tufts University Logo" height="72"
             style="border-radius: 12px; padding: 8px; background: #ffffff;">
      </a>
      <br>
      <sub><strong>Tufts University</strong></sub><br>
      <sub>CubeSat Team</sub><br>
      <sub>Email: <a href="mailto:William.Goldman@tufts.edu">William Goldman</a></sub>
    </td>
    <td align="center" width="260">
      <a href="https://satellite.mit.edu/#/">
        <img src="https://brand.mit.edu/sites/default/files/styles/image_text_2x/public/2023-08/MIT-logo-red-textandimage.png?itok=RNoAwZvy"
             alt="MIT Logo" height="72"
             style="border-radius: 12px; padding: 8px; background: #ffffff;">
      </a>
      <br>
      <sub><strong>MIT</strong></sub><br>
      <sub>Satellite Team</sub><br>
      <sub>Email: <a href="mailto:agarbuz@mit.edu">Aleks Garbuz</a></sub>
    </td>
  </tr>
</table>

<p align="center"><sub>Want your team listed here? <a href="https://github.com/nscheuer/Generalized_ADCS/issues">Open an issue</a> or submit a PR.</sub></p>

