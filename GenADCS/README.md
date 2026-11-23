# GenADCS build and test guide

Run these commands from the repository root. Activate the project virtual
environment first so CMake, Cargo's PyO3 configuration, and pytest use the
project's Python installation:

```bash
source venv/bin/activate
```

## Build the C++ extensions

The C++ backend uses C++17, CMake, Eigen, and pybind11. Configure with testing
enabled, then build the extensions and native test executables:

```bash
cmake -S GenADCS/backends/cpp -B build/cpp -DBUILD_TESTING=ON
cmake --build build/cpp -j2
```

Run all registered C++ tests with CTest:

```bash
ctest --test-dir build/cpp --output-on-failure
```

Build before running CTest. CTest registers the test executables during
configuration, but it cannot run tests whose executables have not been built.

## Build the Rust extensions

Build and copy all Rust extension modules into their Python packages with:

```bash
./GenADCS/backends/rust/build_extensions.sh
```

For a faster debug build, pass `debug`:

```bash
./GenADCS/backends/rust/build_extensions.sh debug
```

## Run the Python tests

After building the native extensions, run all tests under `GenADCS/` with:

```bash
pytest GenADCS/
```

## Run the Rust tests

Rust unit tests are included in seven separate Cargo crates; the root Rust
crate builds the `_simulate` extension and has no unit tests. Run the
test-bearing crates one by one from the repository root:

```bash
export PYO3_PYTHON="$PWD/venv/bin/python"
export PYTHONPATH="$PWD/venv/lib/python3.12/site-packages:$PWD"

(
  set -e
  for crate in \
    state \
    dynamics \
    config \
    helpers/math \
    helpers/math/frames \
    hardware/satellite_geometry \
    hardware/ADCS_subsystem
  do
    cargo test --manifest-path "GenADCS/backends/rust/$crate/Cargo.toml" -- --test-threads=1
  done
)
```

The Python path variables let PyO3-embedded tests import the project's Python
dependencies. The single test thread avoids concurrent NumPy initialization
in those embedded-Python tests. Update the Python 3.12 site-packages path if
the virtual environment uses another Python version.

## Typical full test sequence

Build both native backends before running the Python suite, since some Python
tests import their compiled modules. Then run the native C++ and Rust tests:

```bash
cmake -S GenADCS/backends/cpp -B build/cpp -DBUILD_TESTING=ON
cmake --build build/cpp -j2
./GenADCS/backends/rust/build_extensions.sh

pytest GenADCS/
ctest --test-dir build/cpp --output-on-failure

export PYO3_PYTHON="$PWD/venv/bin/python"
export PYTHONPATH="$PWD/venv/lib/python3.12/site-packages:$PWD"
(
  set -e
  for crate in state dynamics config helpers/math helpers/math/frames \
    hardware/satellite_geometry hardware/ADCS_subsystem; do
    cargo test --manifest-path "GenADCS/backends/rust/$crate/Cargo.toml" -- --test-threads=1
  done
)
```
