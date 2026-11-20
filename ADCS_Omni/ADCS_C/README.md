# ADCS Omni C++ state module

`state/state.hpp` and `state/state.cpp` implement the C++ state types. The
pybind11 extension exposes them as `ADCS_Omni.ADCS_C.state` and provides
`from_python()` / `to_python()` conversion methods for the matching
`ADCS_Omni.ADCS_Python.state` classes. The module is built in place for use
from this source checkout.

## Build

Use Python 3.10 or newer, a C++17 compiler, CMake 3.18 or newer, and
pybind11. Install pybind11 in the Python environment you intend to use:

```bash
python -m pip install 'pybind11>=2.12'
```

From the repository root, configure and build with that same Python:

```bash
cmake -S ADCS_Omni/ADCS_C -B build/adcs_omni_c \
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/adcs_omni_c --parallel
```

CMake writes the extension into `ADCS_Omni/ADCS_C/`. Run Python from the
repository root (or set `PYTHONPATH` to the root):

```python
from ADCS_Omni.ADCS_C.state import AttitudeState, OrbitalState, InternalState, State

state = State(
    AttitudeState([1, 0, 0, 0], [0, 0, 0]),
    OrbitalState.from_cartesian([7000, 0, 0], [0, 7.5, 0], 2451545.0),
    InternalState([0.1]),
)
record = state.to_record()
```

Run the translation-layer check after building:

```bash
python debug/debug_OMNI/debug_state_c_translation.py
```

The C++ `OrbitalState.from_keplerian()` stores the elements, matching the
current Python class. Cartesian conversion of those elements is still
unimplemented, so accessing `position` or `velocity` for such a state raises
an error.
