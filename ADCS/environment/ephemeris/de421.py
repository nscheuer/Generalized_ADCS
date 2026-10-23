__all__ = ["DE421", "Ephemeris"]

import os
import sys
from pathlib import Path
from skyfield.api import load, Loader
from typing import Optional
import numpy as np

from .ephemeris import Ephemeris as EphemerisBase
from ._download import download_with_progress
from ADCS.helpers.asset_manager import AssetManager

class DE421(EphemerisBase):
    KERNEL_FILENAME = "de421.bsp"
    ENV_PATH_VARIABLE = "ADCS_EPHEMERIS_PATH"
    DOWNLOAD_URL = None  # Skyfield's Loader knows the DE421 download URL.
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
        Optional path for the JPL DE421 ephemeris file. If it is missing,
        the kernel is downloaded there. If ``None``, the user cache and
        ``ADCS/environment/kernels/de421.bsp`` are checked before downloading
        to the user cache.
    :type filepath: pathlib.Path | None

    :raises RuntimeError:
        If the ephemeris file cannot be downloaded or loaded.

    .. note::

        This class internally initializes a shared Skyfield
        Skyfield timescale object, which should be reused
        across all time-dependent orbital computations to ensure consistency.

    """

    def __init__(self, filepath: Optional[Path] = None) -> None:
        r"""
        Initialize the ephemeris loader and planetary body objects.

        This constructor loads the JPL DE421 ephemeris file either from a
        user-specified path, the user cache, or the package kernel directory. If the
        file does not exist locally, it is automatically downloaded using
        Skyfield’s loader utilities.

        Upon successful loading, commonly used celestial bodies and a
        timescale object are extracted and stored as attributes.

        :param filepath:
            Optional filesystem path to a DE421 ephemeris file.
            If ``None``, existing local kernels are checked and the user cache
            is used for a download when necessary.
        :type filepath: pathlib.Path | None

        :return:
            ``None``
        :rtype: None

        """
        if filepath is not None:
            path = Path(filepath).expanduser()
            if path.is_file():
                self.planets = load(str(path))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                self.planets = self._download_ephemeris(path)
        else:
            self.planets = load(str(AssetManager().get(self.KERNEL_FILENAME.removesuffix(".bsp"))))

        # Extract common bodies
        self.sun = self.planets['sun']
        self.earth = self.planets['earth']

        # Create a timescale object (shared across all orbital computations)
        self.ts = load.timescale()
        self._closed = False

    @property
    def available_bodies(self) -> tuple[str, ...]:
        r"""List targets and aliases in the loaded DE421 kernel.

        :return: Sorted lower-case body names and aliases.
        :rtype: tuple[str, ...]
        """
        names = self.planets.names()
        return tuple(sorted({alias.lower() for aliases in names.values() for alias in aliases}))

    def position_eci(self, target, j2000, *, observer="earth", apparent=True):
        """Return a body's position relative to an observer in ECI coordinates [km].

        ``target`` and ``observer`` are names or NAIF integer IDs accepted by
        the loaded kernel. J2000 epochs are expressed in Julian centuries.
        Scalar epochs return ``(3,)``; arrays return ``(..., 3)``. Apparent
        positions are computed by default, matching :meth:`sun_position_eci`.

        :param target: Target body name or NAIF identifier.
        :type target: str or int
        :param j2000: Epoch or epochs in Julian centuries since J2000.
        :type j2000: float or numpy.ndarray
        :param observer: Origin body name or NAIF identifier.
        :type observer: str or int
        :param apparent: Apply apparent-position corrections when true.
        :type apparent: bool
        :return: Relative ECI position vector or vectors [km].
        :rtype: numpy.ndarray
        """
        epochs = np.asarray(j2000, dtype=float)
        times = self.ts.tt_jd(epochs * 36525.0 + 2451545.0)
        position = self.planets[observer].at(times).observe(self.planets[target])
        if apparent:
            position = position.apparent()
        position = np.asarray(position.position.km, dtype=float)
        if epochs.ndim == 0:
            return position.reshape(3)
        return np.moveaxis(position, 0, -1)

    def sun_position_eci(self, j2000):
        r"""Find the apparent Earth to Sun position in ECI coordinates.

        :param j2000: Epoch or epochs in Julian centuries since J2000.
        :type j2000: float or numpy.ndarray
        :return: Earth to Sun vector or vectors [km].
        :rtype: numpy.ndarray
        """
        return self.position_eci("sun", j2000)

    def close(self) -> None:
        r"""Close the Skyfield kernel and its BSP file handle.

        :return: None
        :rtype: None
        """
        if not getattr(self, "_closed", True):
            self.planets.close()
            self._closed = True

    def __enter__(self):
        r"""Enter a context that closes the kernel on exit.

        :return: This ephemeris instance.
        :rtype: DE421
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        r"""Close the kernel when leaving a context.

        :param exc_type: Exception type, if one occurred.
        :type exc_type: type or None
        :param exc_value: Exception value, if one occurred.
        :type exc_value: BaseException or None
        :param traceback: Exception traceback, if one occurred.
        :type traceback: traceback or None
        :return: None, allowing an exception to propagate.
        :rtype: None
        """
        self.close()

    def __del__(self):
        r"""Close an unclosed kernel during object finalization.

        :return: None
        :rtype: None
        """
        # Keep existing call sites safe even when they do not yet explicitly
        # close their Ephemeris instance.
        try:
            self.close()
        except Exception:
            pass

    # ----------------------------------------------------------------------
    @staticmethod
    def _packaged_ephemeris_path() -> Path:
        r"""Location of an ephemeris shipped alongside the package, if any.

        Read only. ``de421.bsp`` is 16 MB and is deliberately not shipped in
        the wheel, but a source checkout has one here, and so does an
        installation where a user or sysadmin placed one deliberately.

        :return: Path to ``ADCS/environment/kernels/de421.bsp``.
        :rtype: pathlib.Path
        """
        return Path(__file__).resolve().parents[1] / "kernels" / DE421.KERNEL_FILENAME

    @staticmethod
    def _cache_ephemeris_path() -> Path:
        r"""Per-user cache location for a downloaded ephemeris.

        Honours ``ADCS_EPHEMERIS_PATH`` (a full path to a .bsp file) first,
        then ``XDG_CACHE_HOME``, then the platform default. Deliberately never
        inside ``site-packages``: writing there fails outright on read-only or
        system-managed installs, and any file written survives
        ``pip uninstall`` as an orphan because it is absent from the wheel's
        RECORD.

        :return: Path the ephemeris would be cached at.
        :rtype: pathlib.Path
        """
        override = os.environ.get(DE421.ENV_PATH_VARIABLE)
        if override:
            return Path(override).expanduser()

        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            base = Path(xdg)
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Caches"
        elif sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        else:
            base = Path.home() / ".cache"
        return base / "generalized_adcs" / DE421.KERNEL_FILENAME

    @classmethod
    def _find_existing_ephemeris(cls) -> Path | None:
        r"""First existing ephemeris across the search order, else ``None``.

        Order: explicit ``ADCS_EPHEMERIS_PATH`` / user cache, then the copy
        shipped beside the package.

        :return: An existing ephemeris path, or ``None``.
        :rtype: pathlib.Path | None
        """
        for candidate in (cls._cache_ephemeris_path(), cls._packaged_ephemeris_path()):
            if candidate.exists():
                return candidate
        return None

    def _get_default_ephemeris_path(self) -> Path:
        r"""Where a newly downloaded ephemeris should be written.

        Always the per-user cache, never ``site-packages``.

        :return: Absolute path to the cache location for ``de421.bsp``.
        :rtype: pathlib.Path
        """
        path = self._cache_ephemeris_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ----------------------------------------------------------------------
    def _download_ephemeris(self, save_path: Path):
        r"""
        Download and load the JPL DE421 ephemeris file.

        This method uses Skyfield's loader
        mechanism to retrieve the DE421 ephemeris from official JPL mirrors
        and store it at the specified location. If the file already exists,
        it is reused.

        :param save_path:
            Destination path where the ephemeris file will be stored.
        :type save_path: pathlib.Path

        :return:
            Loaded Skyfield planetary ephemeris kernel.
        :rtype: skyfield.jpllib.SpiceKernel

        :raises RuntimeError:
            If the ephemeris file cannot be downloaded or initialized.

        """
        try:
            loader = Loader(str(save_path.parent))
            url = loader.build_url(self.KERNEL_FILENAME)
            temporary_path = save_path.with_name(save_path.name + ".part")
            download_with_progress(url, temporary_path, self.KERNEL_FILENAME)
            temporary_path.replace(save_path)
            planets = load(str(save_path))
            print(f"✅ Ephemeris downloaded to: {save_path}")
            return planets
        except Exception as e:
            if "temporary_path" in locals():
                temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"❌ Failed to download DE421 ephemeris: {e}") from e


# Import compatibility for existing callers. The implementation lives here.
Ephemeris = DE421
