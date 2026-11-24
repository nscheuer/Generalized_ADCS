set shell := ["bash", "-euo", "pipefail", "-c"]

default: all

# Requires CMake, Ninja, ccache, Eigen, and pybind11 in the active Python environment.
build:
    command -v ninja >/dev/null
    command -v ccache >/dev/null
    cmake -S GenADCS/backends/cpp -B build/genadcs_cpp_ninja -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    cmake --build build/genadcs_cpp_ninja --parallel

test:
    ctest --test-dir build/genadcs_cpp_ninja --output-on-failure --parallel

all: build test
