# Rust tests

**Build.** From the repository root, activate the environment with `source venv/bin/activate`, set `export PYO3_PYTHON="$PWD/venv/bin/python"`, `export PYTHONPATH="$PWD/venv/lib/python3.12/site-packages:$PWD"`, and `export RUSTFLAGS="-D warnings"`, then run `cargo test --manifest-path GenADCS/backends/rust/Cargo.toml --no-run`. Adjust the Python version in `PYTHONPATH` if needed.

**Run tests.** With those variables still set, run `cargo test --manifest-path GenADCS/backends/rust/Cargo.toml -- --test-threads=1` from the repository root. The workspace runs Rust unit tests for the selected state, math, hardware, dynamics, configuration, atmosphere, ephemeris, gravity, and magnetic-field crates, and reuses their build artifacts.
