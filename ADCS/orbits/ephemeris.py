__all__ = ["Ephemeris"]

import hashlib
import os
import sys
from pathlib import Path
from skyfield.api import load, Loader
from skyfield.data import iers
from typing import Optional

#: Planetary kernels this class knows how to fetch by name.
SUPPORTED_EPHEMERIDES = ("de421", "de440s", "de442s")
#: Earth-orientation sources accepted by ``eop``.
SUPPORTED_EOP = (None, "finals2000A")
#: Mirrors of IERS ``finals2000A.all``, tried in order.
FINALS_URLS = (
    "https://maia.usno.navy.mil/ser7/finals2000A.all",
    "https://datacenter.iers.org/products/eop/rapid/standard/finals2000A.all",
)
_DEFAULT_NAME = "de421"

class Ephemeris:
    r"""
    High-precision planetary ephemeris interface based on Skyfield.

    This class provides a unified interface for loading and accessing
    high-precision planetary ephemerides using the
    `Skyfield <https://rhodesmill.org/skyfield/>`_ library. It manages the
    retrieval, storage, and initialization of the JPL DE421 ephemeris file
    and exposes commonly used celestial bodies such as the Earth and the Sun.

    The ephemeris data are represented in the International Celestial
    Reference Frame (ICRF), enabling precise position and velocity
    computations required for orbital dynamics, attitude determination,
    and navigation applications.

    Mathematical Background
    -----------------------
    A planetary ephemeris provides the barycentric position and velocity
    of solar system bodies as continuous functions of time:

    .. math::

        \mathbf{r}_i(t), \; \mathbf{v}_i(t)
        \quad \forall i \in \{\text{Sun, Earth, Moon, planets}\}

    where :math:`\mathbf{r}_i` and :math:`\mathbf{v}_i` are expressed in the
    ICRF frame. These quantities are derived from numerical integration of
    the N-body equations of motion:

    .. math::

        \ddot{\mathbf{r}}_i =
        \sum_{j \neq i} G m_j
        \frac{\mathbf{r}_j - \mathbf{r}_i}
        {\lVert \mathbf{r}_j - \mathbf{r}_i \rVert^3},

    with relativistic and empirical corrections applied in the JPL models.

    The DE421 ephemeris is valid over the time span:

    .. math::

        1900 \le t \le 2050,

    and provides sufficient accuracy for most Earth-orbiting spacecraft
    analyses.

    :param filepath:
        Optional path to a pre-downloaded JPL ephemeris file. If given, it
        takes precedence over ``name``. If ``None``, the kernel named by
        ``name`` is looked up in the user cache, then beside the package, and
        downloaded to the user cache if missing.
    :type filepath: pathlib.Path | None

    :param name:
        Planetary kernel to use when ``filepath`` is ``None``: ``"de421"``
        (default, 16 MB, 1900–2050), ``"de440s"`` (31 MB, 1849–2150, the
        current JPL general-purpose ephemeris) or ``"de442s"`` (31 MB, a DE440
        update mainly for the Uranus system).
    :type name: str

    :param eop:
        Earth-orientation source. ``None`` (default) uses Skyfield's builtin
        UT1 tables and no polar motion. ``"finals2000A"`` downloads the IERS
        ``finals2000A.all`` file to the user cache and installs its UT1 and
        polar-motion tables, which ITRS rotations then include. Polar motion is
        ~0.3 arcsec (~10 m at the surface).
    :type eop: str | None

    :raises RuntimeError:
        If the ephemeris file cannot be downloaded or loaded.

    .. note::

        This class internally initializes a shared Skyfield
        :class:`~ADCS.etc.Ephemeris` timescale object, which should be reused
        across all time-dependent orbital computations to ensure consistency.

    """

    def __init__(
        self,
        filepath: Optional[Path] = None,
        name: str = _DEFAULT_NAME,
        eop: Optional[str] = None,
    ) -> None:
        r"""
        Initialize the ephemeris loader and planetary body objects.

        This constructor loads the JPL DE421 ephemeris file either from a
        user-specified path or from the default project directory. If the
        file does not exist locally, it is automatically downloaded using
        Skyfield’s loader utilities.

        Upon successful loading, commonly used celestial bodies and a
        timescale object are extracted and stored as attributes.

        :param filepath:
            Optional filesystem path to a DE421 ephemeris file.
            If ``None``, a default project-relative path is used.
        :type filepath: pathlib.Path | None

        :return:
            ``None``
        :rtype: None

        """
        if name not in SUPPORTED_EPHEMERIDES:
            raise ValueError(f"name must be one of {SUPPORTED_EPHEMERIDES}, got {name!r}")
        if eop not in SUPPORTED_EOP:
            raise ValueError(f"eop must be one of {SUPPORTED_EOP}, got {eop!r}")
        filename = f"{name}.bsp"

        if filepath is not None:
            # User-provided file path
            self.planets = load(str(filepath))
        else:
            existing = self._find_existing_ephemeris(filename)
            if existing is not None:
                self.planets = load(str(existing))
                print(f"Loaded Ephemeris: {filename}")
            else:
                self.planets = self._download_ephemeris(
                    self._get_default_ephemeris_path(filename)
                )

        # Extract common bodies
        self.sun = self.planets['sun']
        self.earth = self.planets['earth']

        # Create a timescale object (shared across all orbital computations)
        self.eop = eop
        self._eop_path: Optional[Path] = None
        if eop is None:
            self.ts = load.timescale()
        else:
            self.ts = self._load_eop_timescale()
        self._closed = False

    def close(self) -> None:
        """Close the underlying Skyfield kernel and its BSP file handle."""
        if not getattr(self, "_closed", True):
            self.planets.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        # Keep existing call sites safe even when they do not yet explicitly
        # close their Ephemeris instance.
        try:
            self.close()
        except Exception:
            pass

    # ----------------------------------------------------------------------
    def _load_eop_timescale(self):
        r"""Timescale with IERS ``finals2000A`` UT1 and polar motion installed.

        The file is cached next to the ephemeris cache, never in
        ``site-packages``, and is not refreshed automatically: delete it to pick
        up newer values. It is fetched from the first reachable entry of
        :data:`FINALS_URLS`; Skyfield's own IERS address may reject requests.

        :return: Skyfield timescale with a polar-motion table.
        :rtype: skyfield.timelib.Timescale
        """
        cache_dir = self._get_default_ephemeris_path().parent
        loader = Loader(str(cache_dir), verbose=False)
        errors = []
        for url in FINALS_URLS:
            try:
                with loader.open(url, filename="finals2000A.all") as f:
                    finals = iers.parse_x_y_dut1_from_finals_all(f)
                break
            except Exception as e:
                errors.append(f"{url}: {e}")
        else:
            raise RuntimeError("Failed to load IERS finals2000A.all: " + "; ".join(errors))
        # finals2000A.all is now cached, so this reads it instead of downloading.
        ts = loader.timescale(builtin=False)
        iers.install_polar_motion_table(ts, finals)
        self._eop_path = cache_dir / "finals2000A.all"
        return ts

    def provenance(self) -> dict:
        r"""
        Identify the ephemeris and Earth-orientation data in use.

        Intended for result metadata (e.g. visualization logs), so a run can be
        tied to the exact kernel and EOP file that produced it.

        :return:
            ``{"name", "path", "sha256", "eop", "eop_path", "eop_sha256"}``.
            ``eop_*`` entries are ``None`` when builtin tables are used.
        :rtype: dict
        """
        path = Path(self.planets.path)
        eop_path = self._eop_path if self._eop_path is not None and self._eop_path.exists() else None
        return {
            "name": path.stem,
            "path": str(path),
            "sha256": _sha256(path),
            "eop": self.eop,
            "eop_path": None if eop_path is None else str(eop_path),
            "eop_sha256": None if eop_path is None else _sha256(eop_path),
        }

    @staticmethod
    def _packaged_ephemeris_path(filename: str = "de421.bsp") -> Path:
        r"""Location of an ephemeris shipped alongside the package, if any.

        Read only. ``de421.bsp`` is 16 MB and is deliberately not shipped in
        the wheel, but a source checkout has one here, and so does an
        installation where a user or sysadmin placed one deliberately.

        :param filename: Kernel file name, ``de421.bsp`` by default.
        :type filename: str
        :return: Path to ``ADCS/environment/<filename>``.
        :rtype: pathlib.Path
        """
        return Path(__file__).resolve().parents[1] / "environment" / filename

    @staticmethod
    def _cache_ephemeris_path(filename: str = "de421.bsp") -> Path:
        r"""Per-user cache location for a downloaded ephemeris.

        Honours ``ADCS_EPHEMERIS_PATH`` (a full path to a .bsp file) first,
        then ``XDG_CACHE_HOME``, then the platform default. For a kernel other
        than ``de421.bsp``, an ``ADCS_EPHEMERIS_PATH`` override contributes only
        its directory. Deliberately never
        inside ``site-packages``: writing there fails outright on read-only or
        system-managed installs, and any file written survives
        ``pip uninstall`` as an orphan because it is absent from the wheel's
        RECORD.

        :param filename: Kernel file name, ``de421.bsp`` by default.
        :type filename: str
        :return: Path the ephemeris would be cached at.
        :rtype: pathlib.Path
        """
        override = os.environ.get("ADCS_EPHEMERIS_PATH")
        if override:
            path = Path(override).expanduser()
            return path if filename == "de421.bsp" else path.parent / filename

        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            base = Path(xdg)
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Caches"
        elif sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        else:
            base = Path.home() / ".cache"
        return base / "generalized_adcs" / filename

    @classmethod
    def _find_existing_ephemeris(cls, filename: str = "de421.bsp") -> Path | None:
        r"""First existing ephemeris across the search order, else ``None``.

        Order: explicit ``ADCS_EPHEMERIS_PATH`` / user cache, then the copy
        shipped beside the package.

        :param filename: Kernel file name, ``de421.bsp`` by default.
        :type filename: str
        :return: An existing ephemeris path, or ``None``.
        :rtype: pathlib.Path | None
        """
        if filename == "de421.bsp":
            candidates = (cls._cache_ephemeris_path(), cls._packaged_ephemeris_path())
        else:
            candidates = (cls._cache_ephemeris_path(filename), cls._packaged_ephemeris_path(filename))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _get_default_ephemeris_path(self, filename: str = "de421.bsp") -> Path:
        r"""Where a newly downloaded ephemeris should be written.

        Always the per-user cache, never ``site-packages``.

        :param filename: Kernel file name, ``de421.bsp`` by default.
        :type filename: str
        :return: Absolute path to the cache location for ``filename``.
        :rtype: pathlib.Path
        """
        path = self._cache_ephemeris_path() if filename == "de421.bsp" else self._cache_ephemeris_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ----------------------------------------------------------------------
    def _download_ephemeris(self, save_path: Path):
        r"""
        Download and load a JPL ephemeris file.

        This method uses Skyfield’s :class:`~ADCS.etc.Ephemeris` loader
        mechanism to retrieve the DE421 ephemeris from official JPL mirrors
        and store it at the specified location. If the file already exists,
        it is reused.

        :param save_path:
            Destination path where the ephemeris file will be stored. Its file
            name selects the kernel (e.g. ``de421.bsp``).
        :type save_path: pathlib.Path

        :return:
            Loaded Skyfield planetary ephemeris kernel.
        :rtype: skyfield.jpllib.SpiceKernel

        :raises RuntimeError:
            If the ephemeris file cannot be downloaded or initialized.

        """
        try:
            loader = Loader(str(save_path.parent))
            planets = loader(save_path.name)  # downloads if missing
            print(f"✅ Ephemeris downloaded to: {planets.path}")
            return planets
        except Exception as e:
            raise RuntimeError(f"Failed to download {save_path.name} ephemeris: {e}") from e


def _sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
