# C++ tests

**Build.** From the repository root, activate the environment with `source venv/bin/activate`, then run `cmake -S GenADCS/backends/cpp -B build/cpp -DBUILD_TESTING=ON` and `cmake --build build/cpp -j2`. CMake treats compiler warnings as errors by default.

**Run tests.** Run `ctest --test-dir build/cpp --output-on-failure` from the repository root after the build completes.
