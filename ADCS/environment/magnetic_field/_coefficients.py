"""Shared coefficient downloads and spherical harmonic evaluation plumbing."""

from __future__ import annotations

from hashlib import md5
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from ADCS.environment.ephemeris._download import download_with_progress
from ADCS.environment.igrf import _kernel_static


KERNEL_DIR = Path(__file__).resolve().parents[1] / "kernels"


def ensure_kernel(path: Path, url: str, expected_md5: str | None = None) -> Path:
    """Fetch a missing model file into the kernel directory atomically."""
    path = Path(path)
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".part", delete=False
    ) as stream:
        part = Path(stream.name)
    try:
        download_with_progress(url, part, path.name)
        if not part.is_file() or part.stat().st_size == 0:
            raise OSError(f"Downloaded empty magnetic model kernel: {url}")
        if expected_md5 is not None and md5(part.read_bytes()).hexdigest() != expected_md5:
            raise ValueError(f"Downloaded magnetic model kernel failed checksum: {url}")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    return path


def coordinate_dates(r, theta, phi, date):
    """Broadcast coordinates; pair one date or one date per sample."""
    r_a, th_a, ph_a = np.broadcast_arrays(
        np.asarray(r, dtype=float),
        np.asarray(theta, dtype=float),
        np.asarray(phi, dtype=float),
    )
    shape = r_a.shape
    r_f, th_f, ph_f = (
        np.ascontiguousarray(a.reshape(-1)) for a in (r_a, th_a, ph_a)
    )
    if not np.all(np.isfinite(r_f)) or np.any(r_f <= 0):
        raise ValueError("r must contain positive finite geocentric radii in km")
    if not np.all(np.isfinite(th_f)) or np.any((th_f < 0) | (th_f > 180)):
        raise ValueError("theta must contain colatitudes from 0 to 180 degrees")
    if not np.all(np.isfinite(ph_f)):
        raise ValueError("phi must contain finite longitudes")
    try:
        dates = np.atleast_1d(np.asarray(date, dtype="datetime64[ns]"))
    except (ValueError, TypeError) as exc:
        raise TypeError("date must be a datetime or sequence of datetimes") from exc
    if dates.size not in (1, r_f.size):
        raise ValueError(
            f"date has {dates.size} entries; expected 1 or {r_f.size} to match the coordinates"
        )
    if np.any(np.isnat(dates)):
        raise ValueError("date must not contain NaT")
    return shape, r_f, th_f, ph_f, dates


def evaluate_static(r, theta, phi, g, h):
    """Evaluate one fixed coefficient set at all positions, in nT."""
    br, bt, bp = (np.empty(r.size, dtype=float) for _ in range(3))
    _kernel_static(r, theta, phi, g, h, g.shape[0] - 1, br, bt, bp, 512)
    return br, bt, bp


def shaped_field(shape, components):
    if shape == ():
        return tuple(float(c[0]) for c in components)
    return tuple(c.reshape(shape) for c in components)
