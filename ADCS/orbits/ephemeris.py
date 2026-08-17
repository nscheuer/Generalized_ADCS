__all__ = ["Ephemeris"]

import os
import sys
from pathlib import Path
from skyfield.api import load, Loader
from typing import Optional

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
        Optional path to a pre-downloaded JPL DE421 ephemeris file.
        If ``None``, a default location under
        ``ADCS/environment/de421.bsp`` is used.
    :type filepath: pathlib.Path | None

    :raises RuntimeError:
        If the ephemeris file cannot be downloaded or loaded.

    .. note::

        This class internally initializes a shared Skyfield
        :class:`~ADCS.etc.Ephemeris` timescale object, which should be reused
        across all time-dependent orbital computations to ensure consistency.

    """

    def __init__(self, filepath: Optional[Path] = None) -> None:
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
        if filepath is not None:
            # User-provided file path
            self.planets = load(str(filepath))
        else:
            existing = self._find_existing_ephemeris()
            if existing is not None:
                self.planets = load(str(existing))
                print(f"✅ Loaded local ephemeris from {existing}")
            else:
                self.planets = self._download_ephemeris(
                    self._get_default_ephemeris_path()
                )

        # Extract common bodies
        self.sun = self.planets['sun']
        self.earth = self.planets['earth']

        # Create a timescale object (shared across all orbital computations)
        self.ts = load.timescale()

    # ----------------------------------------------------------------------
    @staticmethod
    def _packaged_ephemeris_path() -> Path:
        r"""Location of an ephemeris shipped alongside the package, if any.

        Read only. ``de421.bsp`` is 16 MB and is deliberately not shipped in
        the wheel, but a source checkout has one here, and so does an
        installation where a user or sysadmin placed one deliberately.

        :return: Path to ``ADCS/environment/de421.bsp``.
        :rtype: pathlib.Path
        """
        return Path(__file__).resolve().parents[1] / "environment" / "de421.bsp"

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
        override = os.environ.get("ADCS_EPHEMERIS_PATH")
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
        return base / "generalized_adcs" / "de421.bsp"

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

        This method uses Skyfield’s :class:`~ADCS.etc.Ephemeris` loader
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
            planets = loader("de421.bsp")  # downloads if missing
            print(f"✅ Ephemeris downloaded to: {planets.path}")
            return planets
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download DE421 ephemeris: {e}") from e
