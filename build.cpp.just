set shell := ["bash", "-euo", "pipefail", "-c"]

default: all

# Requires CMake, Ninja, sccache, Eigen, and pybind11 in the active Python environment.
build:
    command -v ninja >/dev/null
    command -v sccache >/dev/null
    python_bin="$(command -v python || command -v python3)"; pybind11_cmake_dir="$($python_bin -m pybind11 --cmakedir)"; cmake -S GenADCS/backends/cpp -B build/genadcs_cpp_ninja -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DPython_EXECUTABLE="$($python_bin -c 'import sys; print(sys.executable)')" -Dpybind11_DIR="$pybind11_cmake_dir"
    cmake --build build/genadcs_cpp_ninja --parallel "$(getconf _NPROCESSORS_ONLN)"

test:
    ctest --test-dir build/genadcs_cpp_ninja --output-on-failure --parallel "$(getconf _NPROCESSORS_ONLN)"

all: build test
