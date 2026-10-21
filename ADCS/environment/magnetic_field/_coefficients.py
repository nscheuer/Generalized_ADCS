"""Shared coefficient downloads and spherical harmonic evaluation plumbing."""

from __future__ import annotations

from hashlib import md5
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from ADCS.environment.ephemeris._download import download_with_progress
from ADCS.helpers.math.spherical_harmonics import _kernel_static


def ensure_kernel(path: Path, url: str, expected_md5: str | None = None) -> Path:
    r"""Fetch a missing magnetic coefficient file with an atomic rename.

    :param path: Destination for the cached coefficient file.
    :type path: pathlib.Path
    :param url: Source URL for a missing file.
    :type url: str
    :param expected_md5: Published checksum, when available.
    :type expected_md5: str or None
    :return: Path to the existing or newly downloaded coefficient file.
    :rtype: pathlib.Path
    :raises ValueError: If a downloaded file fails its checksum.
    """
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
    r"""Broadcast spherical coordinates and align their UTC dates.

    :param r: Geocentric radius [km].
    :type r: float or numpy.ndarray
    :param theta: Geocentric colatitude [deg].
    :type theta: float or numpy.ndarray
    :param phi: East longitude [deg].
    :type phi: float or numpy.ndarray
    :param date: One date or one date per coordinate.
    :type date: datetime.datetime or sequence
    :return: Broadcast shape, flattened coordinates, and aligned date array.
    :rtype: tuple
    :raises ValueError: If coordinates or date counts are invalid.
    """
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
    r"""Evaluate one Schmidt normalized coefficient set at all positions.

    :param r: Flattened geocentric radii [km].
    :type r: numpy.ndarray
    :param theta: Flattened geocentric colatitudes [deg].
    :type theta: numpy.ndarray
    :param phi: Flattened east longitudes [deg].
    :type phi: numpy.ndarray
    :param g: Cosine Gauss coefficients [nT].
    :type g: numpy.ndarray
    :param h: Sine Gauss coefficients [nT].
    :type h: numpy.ndarray
    :return: Flat radial, southward, and eastward components [nT].
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    br, bt, bp = (np.empty(r.size, dtype=float) for _ in range(3))
    _kernel_static(r, theta, phi, g, h, g.shape[0] - 1, br, bt, bp, 512)
    return br, bt, bp


def shaped_field(shape, components):
    r"""Restore broadcast shape to evaluated field components.

    :param shape: Original broadcast coordinate shape.
    :type shape: tuple
    :param components: Flattened radial, southward, and eastward arrays [nT].
    :type components: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :return: Scalar components or arrays in the original shape [nT].
    :rtype: tuple
    """
    if shape == ():
        return tuple(float(c[0]) for c in components)
    return tuple(c.reshape(shape) for c in components)
