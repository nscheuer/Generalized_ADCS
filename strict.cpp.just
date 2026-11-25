set shell := ["bash", "-euo", "pipefail", "-c"]

default: all

cpp_source := "GenADCS/backends/cpp"
strict_build := "build/strict"

# Configure a Clang build and compile the C++ backend without test targets.
configure:
    command -v clang >/dev/null
    command -v clang++ >/dev/null
    command -v ninja >/dev/null
    command -v sccache >/dev/null
    CC=clang CXX=clang++ cmake -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" -S {{cpp_source}} -B {{strict_build}}/clang -G Ninja -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DGENADCS_STRICT_WARNINGS=ON -DGENADCS_EXTENSION_OUTPUT_ROOT="$PWD/{{strict_build}}/clang/extensions"

typecheck: configure
    cmake --build {{strict_build}}/clang --parallel "$(getconf _NPROCESSORS_ONLN)"

# Run selected clang-tidy bug, performance, and readability checks.
lint: typecheck
    command -v clang-tidy >/dev/null
    find {{cpp_source}} -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) -print0 | xargs -0 -r -P "$(getconf _NPROCESSORS_ONLN)" clang-tidy -p {{strict_build}}/clang -checks='bugprone-*,performance-*,readability-*,-readability-magic-numbers' -warnings-as-errors='*'

# Run the Clang Static Analyzer checks provided through clang-tidy.
analyze: typecheck
    command -v clang-tidy >/dev/null
    find {{cpp_source}} -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) -print0 | xargs -0 -r -P "$(getconf _NPROCESSORS_ONLN)" clang-tidy -p {{strict_build}}/clang -checks='clang-analyzer-*' -warnings-as-errors='*'

ubsan:
    command -v clang >/dev/null
    command -v clang++ >/dev/null
    command -v sccache >/dev/null
    cmake -S {{cpp_source}} -B {{strict_build}}/ubsan -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DGENADCS_STRICT_WARNINGS=ON -DGENADCS_EXTENSION_OUTPUT_ROOT="$PWD/{{strict_build}}/ubsan/extensions" -DCMAKE_C_FLAGS='-fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer' -DCMAKE_CXX_FLAGS='-fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer' -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=undefined' -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=undefined'
    cmake --build {{strict_build}}/ubsan --parallel "$(getconf _NPROCESSORS_ONLN)"
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --test-dir {{strict_build}}/ubsan --output-on-failure --parallel "$(getconf _NPROCESSORS_ONLN)"

asan:
    command -v clang >/dev/null
    command -v clang++ >/dev/null
    command -v sccache >/dev/null
    cmake -S {{cpp_source}} -B {{strict_build}}/asan -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DGENADCS_STRICT_WARNINGS=ON -DGENADCS_EXTENSION_OUTPUT_ROOT="$PWD/{{strict_build}}/asan/extensions" -DCMAKE_C_FLAGS='-fsanitize=address -fno-omit-frame-pointer' -DCMAKE_CXX_FLAGS='-fsanitize=address -fno-omit-frame-pointer' -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address' -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=address'
    cmake --build {{strict_build}}/asan --parallel "$(getconf _NPROCESSORS_ONLN)"
    ASAN_OPTIONS=detect_leaks=1 ctest --test-dir {{strict_build}}/asan --output-on-failure --parallel "$(getconf _NPROCESSORS_ONLN)"

tsan:
    command -v clang >/dev/null
    command -v clang++ >/dev/null
    command -v sccache >/dev/null
    cmake -S {{cpp_source}} -B {{strict_build}}/tsan -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DGENADCS_STRICT_WARNINGS=ON -DGENADCS_EXTENSION_OUTPUT_ROOT="$PWD/{{strict_build}}/tsan/extensions" -DCMAKE_C_FLAGS='-fsanitize=thread -fno-omit-frame-pointer' -DCMAKE_CXX_FLAGS='-fsanitize=thread -fno-omit-frame-pointer' -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=thread' -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=thread'
    cmake --build {{strict_build}}/tsan --parallel "$(getconf _NPROCESSORS_ONLN)"
    TSAN_OPTIONS=halt_on_error=1 ctest --test-dir {{strict_build}}/tsan --output-on-failure --parallel "$(getconf _NPROCESSORS_ONLN)"

coverage:
    command -v clang >/dev/null
    command -v clang++ >/dev/null
    command -v llvm-cov >/dev/null
    command -v llvm-profdata >/dev/null
    command -v sccache >/dev/null
    cmake -S {{cpp_source}} -B {{strict_build}}/coverage -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DGENADCS_STRICT_WARNINGS=ON -DGENADCS_EXTENSION_OUTPUT_ROOT="$PWD/{{strict_build}}/coverage/extensions" -DCMAKE_C_FLAGS='-fprofile-instr-generate -fcoverage-mapping' -DCMAKE_CXX_FLAGS='-fprofile-instr-generate -fcoverage-mapping' -DCMAKE_EXE_LINKER_FLAGS='-fprofile-instr-generate' -DCMAKE_SHARED_LINKER_FLAGS='-fprofile-instr-generate'
    cmake --build {{strict_build}}/coverage --parallel "$(getconf _NPROCESSORS_ONLN)"
    mkdir -p {{strict_build}}/coverage/profiles
    rm -f {{strict_build}}/coverage/profiles/*.profraw
    LLVM_PROFILE_FILE="$PWD/{{strict_build}}/coverage/profiles/%p-%m.profraw" ctest --test-dir {{strict_build}}/coverage --output-on-failure --parallel "$(getconf _NPROCESSORS_ONLN)"
    llvm-profdata merge -sparse {{strict_build}}/coverage/profiles/*.profraw -o {{strict_build}}/coverage/coverage.profdata
    mapfile -t objects < <(find {{strict_build}}/coverage -type f \( -name 'cpp_test_*' -o -name '*.so' \) -print | sort)
    test "${#objects[@]}" -gt 0
    extra_objects=()
    for object in "${objects[@]:1}"; do extra_objects+=(-object "$object"); done
    llvm-cov report "${objects[0]}" -instr-profile={{strict_build}}/coverage/coverage.profdata "${extra_objects[@]}"

audit:
    command -v osv-scanner >/dev/null
    osv-scanner scan source -r .

# Sanitizer and coverage recipes remain opt-in; the strict default skips tests.
all: lint analyze audit
