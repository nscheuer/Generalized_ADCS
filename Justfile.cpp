set shell := ["bash", "-euo", "pipefail", "-c"]

default: all

# Apply the C++ formatter to backend and test sources.
format:
    command -v clang-format >/dev/null
    find GenADCS/backends/cpp GenADCS/tests/backends/cpp -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' \) -print0 | xargs -0 -r -P "$(getconf _NPROCESSORS_ONLN)" clang-format -i

# Requires CMake, Ninja, sccache, Eigen, and pybind11 in the active Python environment.
build:
    command -v ninja >/dev/null
    command -v sccache >/dev/null
    cmake -S GenADCS/backends/cpp -B build/genadcs_cpp_ninja -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
    cmake --build build/genadcs_cpp_ninja --parallel

test:
    ctest --test-dir build/genadcs_cpp_ninja --output-on-failure --parallel

all: format build test
