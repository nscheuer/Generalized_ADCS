Create 3.13.2 virtual environment. 

```
python -m venv venv
.\.venv\Scripts\activate
pip install pybind11 cmake numpy
```
# Check cmake and install locally if necessary
Do not run using MSYS2 version of cmake.
```
Get-Command cmake | Select-Object Source
```
If not correct, install Windows cmake using:
```
winget install Kitware.CMake
```

# Install vcpkg
```
cd $env:USERPROFILE
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.bat
```

# Install armadillo & boost
Location: vcpkg/
```
C:\Users\nicla\vcpkg\vcpkg install armadillo
C:\Users\nicla\vcpkg\vcpkg install boost-math:x64-windows
```
This takes almost 15 minutes.

# Build trajectory_planner
```
& "C:\Program Files\CMake\bin\cmake.exe" .. `  -A x64 `  -DCMAKE_TOOLCHAIN_FILE="C:\Users\nicla\vcpkg\scripts\buildsystems\vcpkg.cmake" `  -DVCPKG_TARGET_TRIPLET=x64-windows `  -DPython3_EXECUTABLE="C:\Users\nicla\OneDrive\2ZS9\UROP\Code\Simulator_Python\venv\Scripts\python.exe"

"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release
```