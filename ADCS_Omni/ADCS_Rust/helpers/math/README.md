# Rust mathematical helpers

This package mirrors `ADCS_C/helpers/math` and exposes the same Python import
paths. The `_helpers` and `frames/_frames` extensions are separate PyO3 crates,
matching the two native extension boundaries in the C++ implementation.

Build both extensions in the active Python environment with maturin:

```bash
cd ADCS_Omni/ADCS_Rust/helpers/math
maturin develop
cd frames
maturin develop
```

The Python modules alongside the extensions are the thin import layer; the
vector, dynamics, derivative, rotation, time, and reference-frame kernels run
in Rust. nalgebra handles fixed-size 3×3 matrix operations and 3D vector
algebra, and its full-pivot LU decomposition solves the inertia systems.
Wheel-count-dependent calculations use nalgebra's dynamic matrix and vector
types. NumPy remains the Python-facing array format; the PyO3 NumPy crate
converts nalgebra matrices back to NumPy arrays.
