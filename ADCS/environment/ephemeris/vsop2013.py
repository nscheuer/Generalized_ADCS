"""VSOP2013 Chebyshev ephemeris using the official IMCCE data files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit
from skyfield.api import load

from ._download import download_with_progress, kernel_path
from .ephemeris import Ephemeris


@njit(cache=True, fastmath=True, nogil=True)
def _chebyshev_xyz(coefficients, start: int, count: int, x: float):
    r"""Evaluate three adjacent VSOP Chebyshev series with Clenshaw recursion.

    :param coefficients: Coefficient record containing all three axes.
    :type coefficients: numpy.ndarray
    :param start: First coefficient index for the selected body.
    :type start: int
    :param count: Coefficients per axis.
    :type count: int
    :param x: Normalized interval coordinate in ``[-1, 1]``.
    :type x: float
    :return: Three heliocentric ecliptic coordinates [au].
    :rtype: numpy.ndarray
    """
    result = np.empty(3, dtype=np.float64)
    for axis in range(3):
        base = start + axis * count
        b1 = 0.0
        b2 = 0.0
        for index in range(count - 1, 0, -1):
            b0 = 2.0 * x * b1 - b2 + coefficients[base + index]
            b2 = b1
            b1 = b0
        result[axis] = x * b1 - b2 + coefficients[base]
    return result


_OBLIQUITY = np.deg2rad(23.0 + 26.0 / 60.0 + 21.41136 / 3600.0)
_FRAME_OFFSET = np.deg2rad(-0.05188 / 3600.0)
_C, _S = np.cos(_OBLIQUITY), np.sin(_OBLIQUITY)
_CP, _SP = np.cos(_FRAME_OFFSET), np.sin(_FRAME_OFFSET)
_ECLIPTIC_TO_EQUATORIAL = np.array(
    [[_CP, -_SP, 0.0], [_SP * _C, _CP * _C, -_S], [_SP * _S, _CP * _S, _C]],
    dtype=float,
)


class VSOP2013(Ephemeris):
    """VSOP2013 planetary positions from IMCCE's 32-day Chebyshev tables.

    The selected 1500-year data file is fetched on first query and saved in
    ``ADCS/environment/kernels``. The six source files are each about 392 MB.
    Coordinates are geometric heliocentric solutions, transformed from the
    VSOP2013 dynamical ecliptic J2000 frame into equatorial J2000 coordinates.
    VSOP2013 supplies the Earth-Moon barycenter rather than Earth's center;
    ``earth`` therefore denotes that barycenter in this model.
    """

    BASE_URL = "https://ftp.imcce.fr/pub/ephem/planets/vsop2013/ephemerides/"
    DATA_FILES = (
        (77294.5, 625198.5, "VSOP2013.m4000"),
        (625198.5, 1173102.5, "VSOP2013.m2000"),
        (1173102.5, 1721006.5, "VSOP2013.m1000"),
        (1721006.5, 2268910.5, "VSOP2013.p1000"),
        (2268910.5, 2816814.5, "VSOP2013.p2000"),
        (2816814.5, 3364718.5, "VSOP2013.p4000"),
    )
    BODIES = (
        "mercury", "venus", "earth", "mars", "jupiter", "saturn",
        "uranus", "neptune", "pluto",
    )
    AU_KM = 149597870.7
    _PERIOD_RECORD_BYTES = 23984
    _cache: dict[tuple[str, int], tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]

    def __init__(self, kernel_dir: str | Path | None = None) -> None:
        r"""Choose the directory for downloaded VSOP2013 data files.

        :param kernel_dir: Optional local directory for official data files.
        :type kernel_dir: str or pathlib.Path or None
        :return: None
        :rtype: None
        """
        self.kernel_dir = Path(kernel_dir).expanduser() if kernel_dir is not None else None
        self._cache = {}
        self.ts = load.timescale()

    @property
    def available_bodies(self) -> tuple[str, ...]:
        r"""List supported planets and the Sun.

        :return: Available body names and Earth-Moon barycenter aliases.
        :rtype: tuple[str, ...]
        """
        return self.BODIES + ("sun", "earth-moon barycenter", "emb")

    def _path_for_jd(self, jd: float) -> tuple[float, str]:
        r"""Select the VSOP2013 data file spanning a Julian date.

        :param jd: Julian date to locate.
        :type jd: float
        :return: File start Julian date and data filename.
        :rtype: tuple[float, str]
        :raises ValueError: If the date is outside the available files.
        """
        for start_jd, end_jd, filename in self.DATA_FILES:
            if start_jd <= jd < end_jd:
                return start_jd, filename
        raise ValueError(
            f"VSOP2013 covers JD 77294.5 through 3364718.5; got JD {jd}"
        )

    def _ensure_file(self, filename: str) -> Path:
        r"""Download a missing VSOP2013 file to the kernel directory.

        :param filename: Official data filename.
        :type filename: str
        :return: Path to the available local file.
        :rtype: pathlib.Path
        """
        path = self.kernel_dir / filename if self.kernel_dir is not None else kernel_path(filename)
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_name(path.name + ".part")
            try:
                download_with_progress(self.BASE_URL + filename, temporary_path, filename)
                temporary_path.replace(path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        return path

    def _read_header(self, path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        r"""Read and validate a VSOP2013 file's coefficient layout.

        :param path: Local VSOP2013 data file.
        :type path: pathlib.Path
        :return: First Julian date, body offsets, coefficient counts, and subdivisions.
        :rtype: tuple
        :raises ValueError: If the header or body layout is invalid.
        """
        with path.open("r", encoding="ascii") as stream:
            header = [stream.readline() for _ in range(9)]
        if any(not line for line in header):
            raise ValueError(f"Invalid VSOP2013 file header: {path}")
        if int(header[0].strip()) != 2013:
            raise ValueError(f"Unexpected VSOP2013 file identifier in {path}")
        file_start_jd = float(header[1].strip())
        starts = np.asarray([int(value) for value in header[6].split()], dtype=int)
        ncoef = np.asarray([int(value) for value in header[7].split()], dtype=int)
        nsub = np.asarray([int(value) for value in header[8].split()], dtype=int)
        if starts.shape != (9,) or ncoef.shape != (9,) or nsub.shape != (9,):
            raise ValueError(f"Invalid VSOP2013 coefficient layout in {path}")
        return file_start_jd, starts, ncoef, nsub

    @staticmethod
    def _read_period(path: Path, period_index: int) -> tuple[float, np.ndarray]:
        r"""Read one fixed-length 32-day coefficient period.

        :param path: Local VSOP2013 data file.
        :type path: pathlib.Path
        :param period_index: Zero-based period number within the file.
        :type period_index: int
        :return: Period start Julian date and its Chebyshev coefficients.
        :rtype: tuple[float, numpy.ndarray]
        :raises ValueError: If a record is truncated or malformed.
        """
        # The official ASCII format has 9 header lines, then fixed 23,984-byte
        # interval records: a 23-byte date record and 163 147-byte records.
        with path.open("rb") as stream:
            header_bytes = sum(len(stream.readline()) for _ in range(9))
            stream.seek(header_bytes + period_index * VSOP2013._PERIOD_RECORD_BYTES)
            dates = stream.readline().split()
            if len(dates) < 2:
                raise ValueError(f"Invalid VSOP2013 interval dates in {path}")
            period_start = float(dates[0])
            coefficients = np.empty(978, dtype=float)
            index = 0
            for _ in range(163):
                line = stream.readline()
                if len(line) < 146:
                    raise ValueError(f"Invalid VSOP2013 coefficient record in {path}")
                for column in range(6):
                    offset = 2 + column * 24
                    mantissa = float(line[offset : offset + 20])
                    exponent = int(line[offset + 20 : offset + 24])
                    coefficients[index] = mantissa * (10.0 ** exponent)
                    index += 1
        return period_start, coefficients

    def _period(self, jd: float):
        r"""Load or reuse the coefficient period containing a Julian date.

        :param jd: Query Julian date.
        :type jd: float
        :return: Period start, coefficients, offsets, counts, and subdivisions.
        :rtype: tuple
        """
        file_start, filename = self._path_for_jd(jd)
        key = (filename, int((jd - file_start) // 32.0))
        if key not in self._cache:
            path = self._ensure_file(filename)
            _, starts, ncoef, nsub = self._read_header(path)
            period_start, coefficients = self._read_period(path, key[1])
            self._cache[key] = (period_start, coefficients, starts, ncoef, nsub)
        return self._cache[key]

    def _heliocentric_position_au(self, body_index: int, jd: float) -> np.ndarray:
        r"""Evaluate a body's heliocentric ecliptic position.

        :param body_index: Index in :attr:`BODIES`.
        :type body_index: int
        :param jd: Query Julian date.
        :type jd: float
        :return: Heliocentric ecliptic Cartesian position [au].
        :rtype: numpy.ndarray
        :raises ValueError: If Pluto data are unavailable for the date.
        """
        period_start, coefficients, starts, ncoef, nsub = self._period(jd)
        count = int(ncoef[body_index])
        subdivisions = int(nsub[body_index])
        if body_index == 8 and self._path_for_jd(jd)[1] != "VSOP2013.p2000":
            raise ValueError("VSOP2013 provides Pluto only for years 1500–3000")
        delta = 32.0 / subdivisions
        subinterval = min(int((jd - period_start) / delta), subdivisions - 1)
        interval_start = period_start + subinterval * delta
        x = 2.0 * (jd - interval_start) / delta - 1.0
        start = int(starts[body_index]) - 1 + 6 * count * subinterval
        return _chebyshev_xyz(coefficients, start, count, x)

    @staticmethod
    def _ecliptic_to_equatorial(position_au: np.ndarray) -> np.ndarray:
        r"""Rotate a J2000 ecliptic position into the equatorial frame.

        :param position_au: Ecliptic Cartesian position [au].
        :type position_au: numpy.ndarray
        :return: Equatorial Cartesian position [au].
        :rtype: numpy.ndarray
        """
        return _ECLIPTIC_TO_EQUATORIAL @ position_au

    @staticmethod
    def _body_index(body) -> int | None:
        r"""Resolve a body name or alias to the VSOP planet index.

        :param body: Planet name, Earth-Moon barycenter alias, or Sun.
        :type body: str
        :return: Planet index, or ``None`` for the Sun.
        :rtype: int or None
        :raises ValueError: If the body is not available.
        """
        name = str(body).strip().lower().replace("_", " ").replace("-", " ")
        if name in {"sun", "sol"}:
            return None
        if name in {"earth moon barycenter", "emb"}:
            name = "earth"
        try:
            return VSOP2013.BODIES.index(name)
        except ValueError as exc:
            raise ValueError(f"VSOP2013 does not provide body {body!r}") from exc

    def position_eci(self, target, j2000, *, observer="earth", apparent=False):
        """Return geometric target position relative to observer [km].

        ``j2000`` is in Julian centuries. Since VSOP2013 provides geometric
        Chebyshev positions rather than apparent positions, ``apparent`` is
        accepted for interface compatibility but does not add light-time or
        aberration corrections.

        :param target: Target body name or alias.
        :type target: str
        :param j2000: Epoch or epochs in Julian centuries since J2000.
        :type j2000: float or numpy.ndarray
        :param observer: Origin body name or alias.
        :type observer: str
        :param apparent: Accepted for interface compatibility, without correction.
        :type apparent: bool
        :return: Geometric target relative to observer in ECI coordinates [km].
        :rtype: numpy.ndarray
        """
        target_index = self._body_index(target)
        observer_index = self._body_index(observer)
        epochs = np.asarray(j2000, dtype=float)
        flat_jd = (epochs.reshape(-1) * 36525.0) + 2451545.0
        result = np.empty((flat_jd.size, 3), dtype=float)
        for index, jd in enumerate(flat_jd):
            target_position = (
                np.zeros(3) if target_index is None
                else self._heliocentric_position_au(target_index, float(jd))
            )
            observer_position = (
                np.zeros(3) if observer_index is None
                else self._heliocentric_position_au(observer_index, float(jd))
            )
            result[index] = self._ecliptic_to_equatorial(
                target_position - observer_position
            ) * self.AU_KM
        if epochs.ndim == 0:
            return result[0]
        return result.reshape(epochs.shape + (3,))

    def sun_position_eci(self, j2000):
        r"""Find the geometric Earth to Sun vector in ECI coordinates.

        :param j2000: Epoch or epochs in Julian centuries since J2000.
        :type j2000: float or numpy.ndarray
        :return: Earth to Sun vector or vectors [km].
        :rtype: numpy.ndarray
        """
        return self.position_eci("sun", j2000, observer="earth")

    def close(self) -> None:
        r"""Clear cached VSOP2013 coefficient periods.

        :return: None
        :rtype: None
        """
        self._cache.clear()


__all__ = ["VSOP2013"]
