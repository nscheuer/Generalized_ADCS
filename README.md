# 🚀 Generalized ADCS

<p>
  <a href="https://aeroastro.mit.edu/starlab/">
    <img src="https://raw.githubusercontent.com/nscheuer/Generalized_ADCS/main/docs/source/_static/starlab_logo.svg" alt="STARLab Logo" height="100">
  </a>
  <a href="https://aeroastro.mit.edu/small-satellite-collaborative/">
    <img src="https://raw.githubusercontent.com/nscheuer/Generalized_ADCS/main/docs/source/_static/ssc_logo.png" alt="Small Satellite Collaborative Logo" height="100">
  </a>
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

<p align="center">
  <img src="https://raw.githubusercontent.com/nscheuer/Generalized_ADCS/main/docs/source/_static/release_notes/0_2_2_animation_preview.png"
       alt="ICESat-2 performing attitude maneuvers over Earth at sunset"
       width="75%">
</p>


## ✨ Key Features

<details>
  <summary>✅ Python-first, designed for research and experimentation</summary>

  - Built in Python for accessibility, rapid prototyping, and easy customization.
  - Minimal setup, extensive documentation, and hands-on tutorials.
  - Easy to inspect, debug, modify, and extend with custom algorithms.
  - Numba-accelerated numerical kernels for computationally demanding simulations.
  - Monte Carlo support for statistical analysis and robustness testing.
</details>

<details>
  <summary>✅ Generalized 6-DOF spacecraft dynamics</summary>

  - Attitude propagation using SciPy's <code>solve_ivp</code> with RK45 integration.
  - Fully configurable spacecraft inertia, geometry, and external disturbances.
  - Support for underactuated and overactuated spacecraft configurations.
  - Coming soon: Fast, Numba-accelerated RK4 propagation.
</details>

<details>
  <summary>✅ Flexible orbit propagation</summary>

  - Initialize orbits from position vectors, orbital elements, or TLEs.
  - Keplerian, SGP4, and RK4 orbit propagation.
  - Model atmospheric drag, solar radiation pressure, and J2 perturbations.
</details>

<details>
  <summary>✅ Advanced attitude estimation</summary>

  - Plug-and-play EKF, MEKF, UKF, and SRUKF estimation frameworks.
  - Real-time sensor bias and disturbance estimation.
</details>

<details>
  <summary>✅ Flexible attitude control and optimization</summary>

  - PD, LQR, projection-based, and magnetorquer-only feedback controllers.
  - LP and QP control allocation for underactuated systems.
  - ALTRO-based trajectory optimization.
</details>

<details>
  <summary>✅ Realistic sensor and actuator modeling</summary>

  - Configurable noise and bias models.
  - Sensors: Gyroscopes, magnetometers, sun sensors, star trackers, Earth horizon sensors, and GPS.
  - Actuators: Reaction wheels and magnetorquers.
  - Growing catalog of preconfigured CubeSat-scale hardware models.
  - Explore available models in <code>ADCS/satellite_hardware</code>.
</details>

## ⚡ Performance

Generalized ADCS 3.0 ships three interchangeable backends (Python with Numba, C++ and Rust) behind one API. The charts compare them with Generalized ADCS 2.x, [Basilisk](https://github.com/AVSLab/basilisk) and [Orekit](https://www.orekit.org/) on the same scenario: a torque-free 12 kg spacecraft on a 500 km, 51.6° circular orbit with J2 gravity, fixed-step RK4 at dt = 0.25 s, with the state recorded every step.

<!-- benchmarks:start -->
<!-- Filled in by benchmarks/make_charts.py from benchmarks/results/results.json. -->
<!-- benchmarks:end -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="benchmarks/results/speed-dark.svg">
    <img src="benchmarks/results/speed-light.svg" width="900" alt="Simulation steps per second for each tool in the attitude-plus-orbit and orbit-only scenarios, and final position error after one orbit.">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="benchmarks/results/speedup-dark.svg">
    <img src="benchmarks/results/speedup-light.svg" width="900" alt="Speed-up of each tool over Generalized ADCS 2.x, with the three Generalized ADCS 3.0 backends grouped in a dashed box.">
  </picture>
</p>

<details>
<summary><b>Run time vs. simulated duration, and method</b></summary>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="benchmarks/results/scaling-dark.svg">
    <img src="benchmarks/results/scaling-light.svg" width="820" alt="Wall-clock time against simulated duration on log-log axes for each tool.">
  </picture>
</p>

Speed is the wall-clock time of the propagation call, including each tool's per-step state recording, after an untimed setup and warm-up (median of at least three runs). Accuracy is measured against a reference that integrates the same equations with SciPy's DOP853 at a relative tolerance of 1e-13. Generalized ADCS 3.0 and Basilisk always integrate attitude with the orbit; Orekit has no rigid-body attitude dynamics, so it runs the orbit-only scenario; 2.x integrates attitude with SciPy's adaptive RK45. Reproduce everything with the scripts in [`benchmarks/`](benchmarks).

</details>

## 📦 Installation

```bash
pip install generalized-adcs
```

Optional extras: `viz` (3-D orbit animation), `dev` (test tooling), `docs`
(documentation build), `cpp` (toolchain for the optional C++ add-ons).

```bash
pip install "generalized-adcs[viz]"
```

Requires Python 3.10 or newer. For deterministic reproduction of published
campaign results, see `requirements-repro.txt` in the repository.

Optional add-ons:
- trajectory_planner (tplaunch/pysat) and SALTRO (saltro_py) are optional C++ extensions.
- Build instructions are in docs/Install_WSL.md and docs/Install_Windows.md.



## Papers

- [A Modular Open-Source ADCS Framework for Small Satellite Development and Testing](https://nscheuer.github.io/Generalized_ADCS/papers/package.html) — presented at SmallSat Europe 2026 in Amsterdam, Netherlands.
- [Generalized Attitude Control for Small Spacecraft](https://nscheuer.github.io/Generalized_ADCS/papers/generalized_acs.html) — presented at the 40th Annual Small Satellite Conference, Poster Session 2, in Salt Lake City, Utah.
- [A Feasibility-Aware Attitude Trajectory Planner for Small Satellites](https://nscheuer.github.io/Generalized_ADCS/papers/planner.html) — presented at the 40th Annual Small Satellite Conference, Flash Talks Session 3, in Salt Lake City, Utah.

## 📚 Academic Background

This project is based on the PhD research of **Patrick McKeen**:

- 🔗 Source Code:  
  https://github.com/patrickmckeen/PhD_Dissertation_Code
- 📄 Dissertation:  
  *Computational Methods to Improve Satellite Attitude Determination and Control
  with a Focus on Autonomy, Generalizability, and Underactuation*  
  https://dspace.mit.edu/handle/1721.1/158874
