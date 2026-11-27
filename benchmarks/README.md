# Benchmarks

Speed and accuracy of Generalized ADCS 3.0 (Python, C++ and Rust backends)
against Generalized ADCS 2.x, [Basilisk](https://github.com/AVSLab/basilisk)
and [Orekit](https://www.orekit.org/), on one shared scenario.

## Run it

```bash
bash benchmarks/setup.sh            # once: per-tool environments + 3.0 native builds
python3 benchmarks/run_all.py       # ~5-15 min, writes benchmarks/results/results.json
.bench/genadcs3/bin/python benchmarks/make_charts.py   # SVG charts + tables
```

`python3 benchmarks/run_all.py --quick` is a short smoke test (not for
publishing). Run on a quiet, plugged-in machine; the hardware, OS, tool
versions, date and git commit are stored in `results.json` and printed under
every chart.

## Scenario

| | |
|---|---|
| Spacecraft | 12 kg rigid body, inertia diag(0.13, 0.10, 0.05) kg m², no torques, no wheels |
| Initial attitude | identity, body rates (0.01, −0.015, 0.02) rad/s |
| Orbit | 500 km circular (r = 6,878,137 m), inclined 51.6° |
| Gravity | central + J2 about the inertial z axis; μ = 3.986004418e14 m³/s², R = 6,378,137 m, J2 = 1.08262668e-3 (GenADCS 3.0's values, set in every tool) |
| Integrator | fixed-step classic RK4, dt = 0.25 s |
| Output | full state recorded at every step, by each tool's own mechanism |
| Durations | 1/8, 1/4, 1/2 and 1 orbit (one orbit = 22,708 steps) |

Two scenarios:

* **Attitude + orbit**: torque-free rotation and the J2 orbit, integrated by every tool that supports both.
* **Orbit only**: J2 orbit propagation.

## Method

* **Speed**: wall-clock time of the propagation call (including each tool's per-step recording), after building the simulation and a warm-up run, both untimed. The headline uses one full orbit and the median of at least 3 runs; fast tools repeat until 2 s of samples are collected (max 20 runs). Reported as simulation steps per second.
* **Accuracy**: final position and attitude error against a high-precision reference that integrates exactly the same equations with SciPy's DOP853 at rtol = 1e-13 (`benchmarks/scenario.py`). Attitude error is the rotation angle between the final inertial-to-body DCMs.
* **Isolation**: each tool runs in its own virtual environment (`.bench/<tool>`), launched by `run_all.py` as a subprocess.

## Per-tool notes

| Tool | Version | How it runs the scenario |
|---|---|---|
| Generalized ADCS 3.0 | this branch | `CoupledRK4Propagator` (attitude and orbit together) on the Python (Numba), C++ (pybind11, `-O3`, LTO) or Rust (PyO3, `--release`) backend. The orbit-only scenario runs the same coupled propagator, because 3.0 has no orbit-only numerical J2 propagator. |
| Generalized ADCS 2.x | 0.2.2 (PyPI) | Orbit with its fixed-step RK4 `Orbit` class; attitude with SciPy RK45 (rtol = atol = 1e-7) inside each step, since 2.x has no fixed-step attitude integrator. Earth constants set on the initial `Orbital_State`. Orbit only times the `Orbit` construction, which also evaluates 2.x's environment for every stored state. |
| Basilisk | 2.12.2 (`bsk`) | `Spacecraft` with its default RK4 integrator, J2 through degree-2 spherical harmonics loaded from a coefficient file written from the shared constants; state recorder every step. Basilisk always integrates attitude with the orbit, so orbit only runs the same model. |
| Orekit | 13.1.9 (`orekit-jpype`) | `NumericalPropagator` with `ClassicalRungeKuttaIntegrator` on Cartesian coordinates and `J2OnlyPerturbation`; ephemeris generator records every step. 10,000-step JVM warm-up before timing. Orbit only: Orekit has no rigid-body attitude dynamics. |

## Files

| Path | What |
|---|---|
| `scenario.py` | the shared scenario, constants and reference solution |
| `runners/*_runner.py` | one script per tool, run inside that tool's environment |
| `run_all.py` | orchestrator: runs everything, computes errors, writes `results/results.json` |
| `make_charts.py` | `results.json` → light/dark SVG charts and `table.md` / `table.rst` |
| `setup.sh` | creates `.bench/` environments and builds the 3.0 backends |
