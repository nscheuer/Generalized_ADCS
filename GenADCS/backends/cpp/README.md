# ADCS Omni C++ modules

`state/state.hpp` and `state/state.cpp` implement the aggregate state.
`attitude_state.hpp/.cpp`, `internal_state.hpp/.cpp`, and
`orbital_state.hpp/.cpp` implement the corresponding Python classes.
`state_types.hpp` contains shared fixed-size vector types and
`state_validation.hpp` contains shared finite-value checks. The pybind11
extension exposes them as `GenADCS.backends.cpp.state` and provides
`from_python()` / `to_python()` conversion methods for the matching
`GenADCS.backends.python.state` classes. The frame transforms live under
`helpers/math/frames/` and are exposed from
`GenADCS.backends.cpp.helpers.math.frames`. The remaining math helpers, including
vector operations, quaternion derivatives, rigid-body and wheel dynamics,
quaternion/DCM/MRP/Rodrigues operations, and Julian date/time-scale conversions,
are implemented in Eigen-backed C++ under `helpers/math/` and exposed through
the matching Python package paths. Rotation implementation files follow the
Python module layout: for example, quaternion algebra and conversions are in
`helpers/math/rotations/quaternion/`, while DCM, MRP, and Rodrigues sources sit
beside their matching Python modules. `rotation_types.hpp` holds the shared
Eigen types and class declarations. The hardware subsystem models live under
`hardware/ADCS_subsystem/` with matching C++ and Python filenames. They retain
Python `Bias` and `Noise` effect objects and invoke each object's `apply()`
method through the same effect sequence as the Python implementation. The
satellite geometry models live under `hardware/satellite_geometry/` and mirror
the Python `SatelliteGeometry` and `SatellitePanel` data classes. The
extensions are built in place for use from this source checkout. They cover
state, math helpers and frames, hardware, environment, dynamics, configuration,
propagation, solver, simulation, and satellite modules. The
`environment/` package mirrors the Python environment hierarchy. Planetary
body data, environment composition, atmosphere co-rotation, and J2–J6 gravity
run in C++. The NRLMSISE-00, DE421, IGRF-14, and CHAOS-8 numerical engines
also run in C++ using the bundled model data files. Every non-package Python
environment module has a matching `.cpp` and `.hpp` path under `environment/`.
The `dynamics/` package mirrors the Python force and torque model paths.
Projected-area drag and SRP, gravity-gradient, residual-dipole, and lunisolar
numerical kernels run in the `_dynamics` C++ extension. Zonal force uses the
existing native environment gravity kernel. Model classes retain the Python
`evaluate()` signatures and call native kernels.

## Build

Use Python 3.10 or newer, a C++17 compiler, CMake 3.18 or newer, Ninja,
Eigen 3.3 or newer, and pybind11. CMake uses ccache automatically when it is
installed. Install Eigen through your system package manager (for example,
`libeigen3-dev` on Debian or Ubuntu), then install Ninja, ccache, and pybind11
using your platform package manager and Python environment:

```bash
python -m pip install 'pybind11>=2.12' ninja
```

With the intended Python environment active, configure and build from the C++
source directory:

```bash
cd GenADCS/backends/cpp
cmake --preset ninja
cmake --build --preset ninja
cd ../../..
```

CMake uses ccache as the compiler launcher when `ccache` is available on
`PATH`. If you prefer not to use the preset, add `-G Ninja` to your normal
`cmake -S ... -B ...` configure command; ccache detection remains automatic.

CMake writes `_state` into `GenADCS/backends/cpp/`, `_frames` into
`GenADCS/backends/cpp/helpers/math/frames/`, and `_helpers` into
`GenADCS/backends/cpp/helpers/math/`. It writes `_adcs_subsystem` into
`GenADCS/backends/cpp/hardware/ADCS_subsystem/` and `_satellite_geometry` into
`GenADCS/backends/cpp/hardware/satellite_geometry/`. Run Python from the repository root
(or set `PYTHONPATH` to the root):

```python
from GenADCS.backends.cpp.state import AttitudeState, OrbitalState, InternalState, State

state = State(
    AttitudeState([1, 0, 0, 0], [0, 0, 0]),
    OrbitalState.from_cartesian([7000, 0, 0], [0, 7.5, 0], 2451545.0),
    InternalState([0.1]),
)
record = state.to_record()

from GenADCS.backends.cpp.helpers.math import frames
body_vector = frames.inertial_to_body([1, 0, 0], [1, 0, 0, 0])

from GenADCS.backends.cpp.hardware.ADCS_subsystem import ADCSSubsystem, ReactionWheel
wheel = ReactionWheel((0, 0, 0), (0, 0, 1), 0.04, 0.2, 5.0)
subsystem = ADCSSubsystem(reaction_wheels=[wheel])

from GenADCS.backends.cpp.hardware.satellite_geometry import SatelliteGeometry, SatellitePanel
geometry = SatelliteGeometry(12.5, [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
                             panels=[SatellitePanel((1, 0, 0), (0, 0, 1), (2, 1))])
```

Run the translation-layer check after building:

```bash
python debug/debug_OMNI/debug_state_c_translation.py
python debug/debug_OMNI/debug_frames_c_translation.py
python debug/debug_OMNI/debug_helpers_c_translation.py
python debug/debug_OMNI/debug_adcs_subsystem_c_translation.py
python debug/debug_OMNI/debug_satellite_geometry_c_translation.py
python debug/debug_OMNI/debug_environment_c_translation.py
```

The C++ `OrbitalState.from_keplerian()` stores the elements, matching the
current Python class. Cartesian conversion of those elements is still
unimplemented, so accessing `position` or `velocity` for such a state raises
an error.

## Keplerian + RK4 simulation loop

`simulate.py` calls the native `simulate_separate_rk4` entry point once.
Its binding converts configuration and releases the GIL; the native loop
advances attitude and orbit, samples sensors at subsystem ticks, updates the
persistent EKF or MEKF state, and records the requested output before advancing
again. Initial sensor and estimator sampling occurs at time zero. A final
partial propagation step does not introduce an extra sensor tick.

`SimulationSampler` prepares sensor layout, estimator configuration, and reusable
sample buffers once. Sensor random streams use the subsystem tick and sensor
index, independently of the result recording stride. Estimation continues at
every subsystem tick even when that tick is not recorded. Rows outside sensor
ticks retain NaN sensor and estimator diagnostics.

Progress is written by native C++ after the complete step, including sensor and
estimator work. Python assembles the returned result arrays; it does not replay
sensor or estimator sampling for this path. The native propagation-only entry
points remain available for callers that do not need simulation sampling.

## Coupled simulation loop

Coupled RK4 and DOP853 simulations also sample sensors and update persistent
estimator state inside native integration, with the GIL released. DOP853 stops
at the common recording/sensor grid; rejected trial steps do not sample sensors
or advance the estimator. Native progress uses the same completed/total and
step/s display as Keplerian + RK4 and includes sampling and estimation work.
Python performs result assembly after the native simulation returns.

## MISRA C++:2023

CMake selects ISO C++17, the language version targeted by MISRA C++:2023. The
native state implementation uses explicit constructors, fixed-size arrays for
fixed-dimension vectors, checked indexing, named types, and finite-value
validation. The pybind11 adapter necessarily uses Python objects, dynamic
conversion, and exception translation; those interfaces need documented
project deviations and are not covered by a formal MISRA compliance claim.
The frame transforms reproduce the Python implementation, including its
low-order IAU 1976 precession, four-term nutation, GMST82, and WGS84 geodetic
model. No MISRA static-analysis tool has been run on these modules.
