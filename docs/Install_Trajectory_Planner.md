# Install trajectory_planner

`trajectory_planner` is the optimization module that computes a reference attitude/control trajectory for hard pointing maneuvers, which is then tracked by ADCS controllers.

For algorithm details and design background, see: https://nscheuer.github.io/SALTRO

## Windows

### Prerequisites
- Visual Studio with C++ workload
- Python 3.13+
- Windows CMake (not MSYS CMake)
- vcpkg

### Ensure Windows CMake is selected
```powershell
Get-Command cmake | Select-Object Source
```

If needed:
```powershell
winget install Kitware.CMake
```

### Install vcpkg dependencies
```powershell
$env:USERPROFILE\vcpkg\vcpkg install armadillo
$env:USERPROFILE\vcpkg\vcpkg install boost-math:x64-windows
```

### Build trajectory_planner
```powershell
cd trajectory_planner
mkdir build
cd build

$CMAKE_EXE = "C:\Program Files\CMake\bin\cmake.exe"
$VCPKG_ROOT = "$env:USERPROFILE\vcpkg"
$PYTHON_VENV = "$PWD\..\..\venv\Scripts\python.exe"

& $CMAKE_EXE .. `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DPython3_EXECUTABLE="$PYTHON_VENV"

& $CMAKE_EXE --build . --config Release
```

## Linux / WSL

### Install system dependencies
```bash
sudo apt update
sudo apt install -y cmake g++ libarmadillo-dev libboost-math-dev
```

### Build trajectory_planner
```bash
cd trajectory_planner
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=$(which python3) \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DPYTHON_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
  -DPYTHON_INCLUDE_DIRS=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
  -DPYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.12.so \
  -DPYTHON_LIBRARIES=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.12.so

make -j$(nproc)
```

## Run trajectory planner examples

- Tutorial 06 script: [examples/tutorials/06_trajectory_planner.py](../examples/tutorials/06_trajectory_planner.py)
- SALTRO tutorial script: [examples/tutorials/07_SALTRO.py](../examples/tutorials/07_SALTRO.py)

Run from repository root with your virtual environment active:

```bash
python examples/tutorials/06_trajectory_planner.py
python examples/tutorials/07_SALTRO.py
```

## SALTRO installation and usage

SALTRO is a separate optional module.

- SALTRO docs site: https://nscheuer.github.io/SALTRO
- SALTRO install page in this repository: [Install_SALTRO.md](Install_SALTRO.md)
