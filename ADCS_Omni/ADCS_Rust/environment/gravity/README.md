# Rust zonal gravity

This crate mirrors the Python J2–J6 gravity model. It evaluates the central term and selected zonal harmonics directly in Rust, with the same Earth gravitational parameter, reference radius, and coefficient convention. The Python package keeps one wrapper and one Rust source file for each model file.

Build the extension from this directory with `maturin develop`, or validate its Rust code with `cargo check`.
