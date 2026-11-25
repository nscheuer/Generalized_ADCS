set shell := ["bash", "-euo", "pipefail", "-c"]

default: format

# Apply clang-format to C++ backend and test sources in parallel.
format:
    command -v clang-format >/dev/null
    find GenADCS/backends/cpp GenADCS/tests/backends/cpp -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' \) -print0 | xargs -0 -r -P "$(getconf _NPROCESSORS_ONLN)" clang-format -i
