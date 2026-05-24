# Prerequisites
- macOS (Apple Silicon or Intel)
- [Xcode Command Line Tools](https://mac.install.guide/commandlinetools/) (provides `clang`/`clang++`): `xcode-select --install`
- [Homebrew](https://brew.sh)
- Python **3.10+** via [pyenv](https://github.com/pyenv/pyenv) (the macOS *system* Python is typically 3.9.x, which is too old for this project — `requires-python >= 3.10`)

> Note: this project is officially documented for Windows / WSL / Linux. macOS is community-supported; the steps below are verified on Apple Silicon (Homebrew prefix `/opt/homebrew`). On Intel Macs substitute the Homebrew prefix `/usr/local` wherever `/opt/homebrew` appears.

# What is trajectory_planner?
`trajectory_planner` is the optimization module that computes reference attitude/control trajectories for difficult pointing maneuvers.

For more algorithm background, see https://nscheuer.github.io/SALTRO.

# macOS Setup

Install the toolchain:
```bash
xcode-select --install                 # if not already installed
brew install pyenv armadillo boost
```
`armadillo` and `boost` are required to build `trajectory_planner`. `cmake` is **not** needed system-wide — it is installed as a pinned pip dependency into the virtual environment.

Install and select a supported Python:
```bash
pyenv install 3.12.13
```
(You do not need to change your global Python; the venv is created directly from this interpreter below.)

# Instructions
## Create Virtual Environment
Inside the `Generalized_ADCS/` folder:
```bash
~/.pyenv/versions/3.12.13/bin/python3 -m venv venv
source venv/bin/activate
python --version            # should print 3.12.13
```
Install Python dependencies using pip:
```bash
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install git+https://github.com/jcrudy/choldate.git --no-build-isolation
pip install -e . --no-deps --no-build-isolation
```
`choldate` must be installed with `--no-build-isolation` because its compilation depends on pip having access to Cython (already installed from `requirements.txt`). The final line registers the `ADCS` package itself in editable mode.

## Build trajectory_planner
`trajectory_planner` installation is documented in one canonical page:

- [Install_Trajectory_Planner.md](Install_Trajectory_Planner.md)

Use the **macOS** section there for the dependency and build commands.

## Optional: Build SALTRO (saltro_py)
SALTRO installation is documented in one canonical page:

- [Install_SALTRO.md](Install_SALTRO.md)

`OldPlanner/` and `SALTRO/` are both included as git submodules in this
repository. If you did not clone with `--recurse-submodules`, initialise them
before building either optional add-on:

```bash
git submodule update --init --recursive OldPlanner SALTRO
```

The canonical SALTRO page does not yet cover macOS. The verified macOS build, run with the virtual environment active and from the repo root, is:
```bash
cd SALTRO && mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DSALTRO_WARNINGS_AS_ERRORS=OFF \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DPython_ROOT_DIR="$VIRTUAL_ENV" \
  -DPython_FIND_VIRTUALENV=ONLY
cmake --build . -j$(sysctl -n hw.logicalcpu)
```
Two macOS-specific notes:
- `-DSALTRO_WARNINGS_AS_ERRORS=OFF` is required: SALTRO's default `-Werror -Wconversion -Wshadow` does not pass under Apple `clang`.
- The `Python_ROOT_DIR` / `Python_FIND_VIRTUALENV=ONLY` hints are required because the bundled pybind11 (v2.12.0) otherwise silently selects the Xcode framework Python 3.9 and produces a `cpython-39` module the venv cannot import. Verify the artifact is `SALTRO/build/saltro_py.cpython-312-darwin.so`.

### Known caveat
`pysat` (from `trajectory_planner`) and `saltro_py` both register a pybind11 class named `Satellite` in the process-global type registry, so importing **both** into the same Python process raises `generic_type: type "Satellite" is already registered`. Each works correctly on its own and via the normal `ADCS.controller` entry points; only co-import in one process is affected.

## Debugging tplaunch
macOS uses LLDB (not GDB):
```bash
lldb -- python examples/tutorials/06_trajectory_planner.py
```
