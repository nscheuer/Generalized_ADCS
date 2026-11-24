# Rust tests and Python extensions

Run these commands from the repository root with the project virtual environment
active. The Rust unit tests and the Python-importable PyO3 extensions are separate
builds: `cargo test` does not install the extension modules used by pytest.

```bash
source venv/bin/activate
export PYO3_PYTHON="$PWD/venv/bin/python"
export PYTHONPATH="$PWD/venv/lib/python3.12/site-packages:$PWD"
export RUSTFLAGS="-D warnings"
```

Adjust `python3.12` in `PYTHONPATH` to match the virtual environment's Python
version if needed.

## Clean Rust build and unit tests

To rebuild Rust from scratch, remove prior Cargo artifacts and copied extension
libraries, then build and run the Rust unit tests:

```bash
cargo clean --manifest-path GenADCS/backends/rust/Cargo.toml
rm -f GenADCS/backends/rust/_*.so
find GenADCS/backends/rust -mindepth 2 -name '_*.so' -delete
cargo test --manifest-path GenADCS/backends/rust/Cargo.toml -- --test-threads=1
```

The workspace tests cover the selected state, math, hardware, dynamics,
configuration, atmosphere, ephemeris, gravity, and magnetic-field crates.

## Build PyO3 extensions

Build the importable Python modules separately:

```bash
bash GenADCS/backends/rust/build_extensions.sh
python -c 'from GenADCS.backends.rust.state import State; from GenADCS.backends.rust.dynamics import _dynamics; import GenADCS.backends.rust._simulate; print("Rust extension imports OK")'
```

The script uses Cargo's release profile and copies each shared library beside
its Python package. Use `bash` because the script may not have its executable
bit set. If a module beginning with `_` cannot be imported, check the build
output and run the script again.

## Python tests

The backend-only Python tests need no native extensions:

```bash
python -m pytest GenADCS/tests/backends/python
```

Binding, conformance, differential, and full-suite tests import native backends.
Build both Rust and C++ extensions before running them. For the C++ prerequisites
and build commands, see [README_CPP.md](README_CPP.md). With both backends built,
run the complete suite from the repository root:

```bash
python -m pytest GenADCS/tests
```
