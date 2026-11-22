# Rust DE421 ephemeris

This package parses the bundled DE421 DAF/SPK kernel in Rust and evaluates its type 2 and type 3 Chebyshev segments directly. It does not call the C++ implementation or Skyfield. `de421/de421.bsp` is included in the Python distribution.

The Rust API mirrors `ADCS_Python.environment.ephemeris`: positions are Earth-centered km, velocities are Earth-centered km/s, and input epochs are TDB Julian dates. It supports scalar and NumPy-shaped epoch inputs for `position_eci` and `velocity_eci`, named and NAIF body identifiers, state queries, and eclipse geometry.

Build the extension from this directory with `maturin develop` or `cargo check`.
