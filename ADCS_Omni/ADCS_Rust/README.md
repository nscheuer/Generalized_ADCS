# ADCS Omni Rust extensions

This package contains Rust implementations of selected ADCS Omni components,
exposed to Python through PyO3. Python modules import compiled extension
libraries (for example, `state._state` and `solver._solver`); the `.rs` source
files cannot be imported directly.

## Requirements

- Python 3.10 or newer, with the project environment active
- Rust and Cargo (installable with [rustup](https://rustup.rs/))
- A C linker and the platform's standard build tools

The Rust crates use PyO3 0.20.3. The build script invokes Cargo for each
extension crate and copies the resulting shared libraries beside their Python
packages. Run it from any working directory:

```bash
./ADCS_Omni/ADCS_Rust/build_extensions.sh
```

The default build uses Cargo's optimized `release` profile. For a faster,
unoptimized development build, pass `debug`:

```bash
./ADCS_Omni/ADCS_Rust/build_extensions.sh debug
```

The script builds the state, math helpers and frames, hardware, environment,
dynamics, configuration, solver, coupled propagator, and simulation extensions.
Build output is placed in each crate's `target/` directory; the importable
libraries are copied into their corresponding package directories under
`ADCS_Omni/ADCS_Rust/`.

After building, run Python from the repository root so the `ADCS_Omni` package
is importable. For example:

```bash
python -c 'from ADCS_Omni.ADCS_Rust.state import State; print(State)'
```

If an import fails with `ModuleNotFoundError` for a name beginning with `_`,
build the extensions again with the script above. Compilation errors are
reported by Cargo; check that the Rust toolchain and native linker are
installed and available on `PATH`.
