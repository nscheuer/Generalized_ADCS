# Repository guide

## Structure

- `GenADCS/backends/python/`, `cpp/`, and `rust/` contain the three implementations. Each follows the same domain layout: state, helpers, environment, dynamics, hardware, propagation, solver, and simulation. `backends/shared/` contains result recording and plotting shared at the Python interface.
- `GenADCS/tests/` is divided by testing purpose (`backends`, `bindings`, `conformance`, `differential`) and then by domain/module. Keep a new test beside the suite it belongs to, following the matching source hierarchy.
- C++ sources and CMake configuration live in `GenADCS/backends/cpp/`. Rust crates live in `GenADCS/backends/rust/`, often one crate per module, with workspace configuration at its root.
- `build.python.just`, `build.cpp.just`, and `build.rust.just` define the corresponding build and test workflows.

## Python backend: keep numerical execution in Numba

Python is the reference implementation. Keep repeated numerical work, especially simulation and propagation loops, fully in Numba-compiled kernels (`@njit`, normally `cache=True`; use `nogil=True` where appropriate). Prepare Python objects, validate inputs, allocate outputs, and convert results outside the kernel; pass kernels plain numeric arrays, scalars, and Numba-compatible typed data. Do not put Python callbacks, object methods, dataclasses, or per-step Python orchestration inside a hot loop. A Python loop around a compiled step is not equivalent to one compiled integration loop and can lose most of the performance benefit. Keep numerical behavior and sampling semantics consistent across backends.

## C++ backend: the source quartet

For a Python-exposed C++ module, follow the module quartet:

1. `name.hpp`: native declarations and types.
2. `name.cpp`: native implementation.
3. `name_py.cpp`: pybind11 boundary that converts inputs, calls the native API, and exposes results.
4. `name.py`: Python package shim that imports/re-exports the compiled extension API.

Register native implementation files, binding files, and extension targets in `GenADCS/backends/cpp/CMakeLists.txt`; follow the module naming and output path already used nearby. Some features split declarations or bindings across multiple files, but preserve the same boundary. Production `.hpp`/`.cpp` code must remain fully C++: no Python, NumPy, or pybind11 types or calls. The numerical algorithm and its integration loop belong in native C++; Python should only prepare/validate inputs and invoke the extension, not run a per-step numerical loop. Native C++ tests call the C++ API directly. Python binding tests may test conversion behavior separately. Follow `GenADCS/tests/backends/cpp/AGENTS.md` for the detailed native-test rules.

## Rust backend

Rust modules are usually small crates: `name.rs` (or module files) contains the Rust implementation, `name_py.rs` contains PyO3 bindings, and `Cargo.toml`/`pyproject.toml` configure the crate and Python extension. The crate `lib.rs` declares modules and exports the extension. Keep numerical work and repeated simulation loops fully in Rust; PyO3 is the boundary for argument conversion and Python exposure. Do not send each integration step through Python or implement the numerical core as a Python wrapper around Rust single-step calls. Register workspace members and extension build/install behavior in the existing Rust workspace scripts. Rust unit tests should exercise Rust APIs directly; Python-facing conversion belongs in binding tests.

## Tests and how they work

- `GenADCS/tests/backends/python/`: Python backend behavior and numerical implementation tests.
- `GenADCS/tests/backends/cpp/`: native C++ tests, generally compiled and run by CTest; these should exercise production C++ APIs directly, without starting Python.
- `GenADCS/tests/backends/rust/`: Rust backend integration or Python-facing backend tests; Rust unit tests also live with their crates and run through Cargo.
- `GenADCS/tests/bindings/`: tests Python-to-native conversion and the exposed extension API. These intentionally cross the binding boundary.
- `GenADCS/tests/conformance/`: checks common public behavior against shared expectations across backends.
- `GenADCS/tests/differential/`: compares backend results for matching inputs, useful for numerical parity and regression checks.

Directories within these suites mirror the domain modules (for example `dynamics/zonal_model` or `helpers/math/rotations`). `conftest.py` files provide scoped fixtures and parameterization; keep fixtures near the tests that use them and reuse shared fixtures from parent conftests. Python suites use pytest. Native C++ tests are registered in CMake and run as CTest executables. Rust unit tests use Cargo; Rust extensions are built/imported for Python-level Rust suites. Use the repository commands in the `build.*.just` files: Python runs pytest over `GenADCS/tests`, C++ runs CTest after CMake builds, and Rust runs workspace Cargo tests plus its configured Python extension environment. Add tests to the appropriate layer: native algorithms to native tests, conversion contracts to binding tests, and cross-backend behavior to conformance/differential tests.
