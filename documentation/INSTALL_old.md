# Prerequisities
- Linux or WSL Machine

# WSL Setup
To install WSL onto on a Windows machine, open Powershell in Administrator Mode.
```powershell
wsl --install -d Ubuntu
```
After installation, Windows will open a new terminal window. Here, enter a `username` and `password`. Remember the `password`, as it is required to run `sudo` commands in the WSL Shell. After this is complete, you should have a working Linux shell.
```bash
user@machine:~$
```

## Connect WSL to VS Code
Open VS Code in Windows and install the [Remote-WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) extension. Now, you can open WSL using VS Code by selecting the blue arrow button in the bottom left of the window and selecting 'Connect to WSL'.

### Note about filesystems
When working with WSL, it is recommended to have projects stored/cloned on the WSL partition of your PC, rather than in the Windows filesystem. When opening File Explorer, you may note a new folder at the very bottom of the left pane, called 'Linux'. A recommended location for projects may be 'Linux/Ubuntu/home/user/Documents'. 

Running WSL on a Windows filepath is very slow, since Windows uses a NTFS filesystem, while WSL works with ext4 images, with a virtualization layer between. This layer can make certain operations extremely slow, such as `pip install ...` and `cmake ...`.

# Instructions
## Create Virtual Environment
Ensure that Python 3.x is installed in your WSL distro. This can be checked by opening a WSL terminal and running 'python3 --version'.
Create a virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate
```
Install Python dependencies using pip:
```bash
pip install -r requirements.txt
pip install git+https://github.com/jcrudy/choldate.git --no-build-isolation
```
The reason choldate has to be installed with --no-build-isolation is that its compilation depends on pip having access to Cython.

## Build trajectory_planner
### Install required C++ packages
```bash
sudo apt install -y cmake g++ libarmadillo-dev libboost-math-dev
```

# Build tplaunch
```bash
cmake ../.. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=$(which python3) \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DPYTHON_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
  -DPYTHON_INCLUDE_DIRS=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
  -DPYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.12.so \
  -DPYTHON_LIBRARIES=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.12.so

make -j$(nproc)
```

# Install subpackages
```
cd Generalized/ADS/ADCS
pip install -e .
cd ../control
pip install -e .
cd ../estimation
pip install -e .
cd ../helpers
pip install -e .
cd ../orbital_state
pip install -e .
cd ../satellite_hardware
pip install -e .
```

# Install de421.bsp
Run the Python file 'download_be421.py'

# Install choldate
```
pip install git+https://github.com/jcrudy/choldate.git --no-build-isolation
```

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