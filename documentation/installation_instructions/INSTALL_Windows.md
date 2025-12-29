# Prerequisites
- C++, preferably using [Visual Studio](https://learn.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2022)
- Python 3.13, with Python executable on system [PATH](https://realpython.com/add-python-to-path/)

# Instructions
## Create Virtual Environment
### Ensure correct Python version
Ensure that the version of Python being used in Powershell is the correct one (13.x+). Do this by running:
```powershell
python --version
```
In case Powershell does not display your installed version of Python (13.x+), but you are confident it is installed on your machine, run Python (13.x+) explicitly using:
```powershell
py -3.13 --version
```

### Virtual environment
Inside the `Generalized_ADCS/` folder run in Powershell:
```powershell
python -m venv venv
.\.venv\Scripts\activate
```

### Install Python dependencies using pip
In the same terminal, run:
```powershell
pip install -r requirements.txt
pip install git+https://github.com/jcrudy/choldate.git --no-build-isolation
```
The reason choldate has to be installed with --no-build-isolation is that its compilation depends on pip having access to Cython.

## Build trajectory_planner
`trajectory_planner` is a C++ project that is imported into the Python codebase using Pybind and must be built from source. 
### Ensure correct CMake executable
The C++ project only compiles with Windows CMake, not the default MSYS CMake, since the latter is hardcoded to use MSYS Python, rather than our virtual environment. Check which version of CMake exists on your machine:
```powershell
Get-Command cmake | Select-Object Source
```
In case that MSYS CMake is selected, i.e. the output is `C:\msys54\mingw64\bin\cmake.exe`, please install Windows CMake and use it to run future compilations as follows:
```powershell
winget install Kitware.CMake
```
This should install Windows CMake to the path `C:\Program Files\CMake\bin\cmake.exe`. Use this executable for compilation below.

### Install vcpkg
Check if vcpkg is installed and on PATH:
```powershell
Get-Command vcpkg | Select-Object Source
```
If it is not installed, it may be installed using the following commands:
```powershell
cd $env:USERPROFILE
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.bat
```

### Install C++ dependencies using vcpkg
Location: vcpkg/
```powershell
$env:USERPROFILE\vcpkg\vcpkg install armadillo
$env:USERPROFILE\vcpkg\vcpkg install boost-math:x64-windows
```
This takes almost 15 minutes.

### Build trajectory_planner
```powershell
$CMAKE_EXE = "C:\Program Files\CMake\bin\cmake.exe"
$VCPKG_ROOT = "$env:USERPROFILE\vcpkg"
$PYTHON_VENV = "$PSScriptRoot\..\Simulator_Python\venv\Scripts\python.exe"

# Configure the CMake build
& $CMAKE_EXE .. `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DPython3_EXECUTABLE="$PYTHON_VENV"

# Build in Release mode
& $CMAKE_EXE --build . --config Release
```