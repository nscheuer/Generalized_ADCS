# Rust atmosphere model

This extension provides the `ADCS_Python.environment.atmosphere` API in Rust. The NRLMSISE-00 numerical model is supplied by the pure Rust `nrlmsise00` crate (MIT licensed); this package owns the PyO3 bindings, inertial/geodetic input conversion, and ADCS defaults. The model crate reports bit-for-bit agreement with the reference C implementation for its `f64` path. Its tables and numerical routines are compiled as Rust code rather than called through a C ABI.

The wrapper uses the Rust-backed `ADCS_Rust.helpers.math.time` and `ADCS_Rust.helpers.math.frames` conversions for UTC calendar fields, UTC-to-TT, and GCRF-to-geodetic position conversion. Build the helpers before using inertial-position density calls.

Build the extension from this directory with `maturin develop` or `cargo check`.
