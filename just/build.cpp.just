set shell := ["bash", "-euo", "pipefail", "-c"]

default: all

# Requires CMake, Ninja, sccache, Eigen, and pybind11 in the active Python environment.
build:
    command -v ninja >/dev/null
    command -v sccache >/dev/null
    cmake -S GenADCS/backends/cpp -B build/genadcs_cpp_ninja -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
    cmake --build build/genadcs_cpp_ninja --parallel "$(getconf _NPROCESSORS_ONLN)"

test:
    ctest --test-dir build/genadcs_cpp_ninja --output-on-failure --parallel "$(getconf _NPROCESSORS_ONLN)"

all: build test
