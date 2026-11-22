"""Rust-backed magnetic field model interface and harmonic kernel."""
from ._magnetic_field_model import MagneticFieldModel, magnetic_field_inertial_tesla


def read_shc(path, nmin, nmax):
    """Read an SHC coefficient file through the native Rust parser."""
    from pathlib import Path
    from ._magnetic_field_model import read_shc as _read_shc
    return _read_shc(str(Path(path)), nmin, nmax)


__all__ = ["MagneticFieldModel", "magnetic_field_inertial_tesla", "read_shc"]
