__all__ = ["Ephemeris"]

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
            # Default: search or download under project_root/ADCS/environment/de421.bsp
            default_path = self._get_default_ephemeris_path()
            if default_path.exists():
                self.planets = load(str(default_path))
                print(f"✅ Loaded local ephemeris from {default_path}")
            else:
                self.planets = self._download_ephemeris(default_path)

        # Extract common bodies
        self.sun = self.planets['sun']
        self.earth = self.planets['earth']
        self.moon = self.planets['moon']

        # Create a timescale object (shared across all orbital computations)
        self.ts = load.timescale()

    # ----------------------------------------------------------------------
    def _get_default_ephemeris_path(self) -> Path:
        r"""
        Determine the default local path for the DE421 ephemeris file.

        This method constructs the expected filesystem location for storing
        the ephemeris file relative to the project root. If the parent
        directory does not exist, it is created automatically.

        :return:
            Absolute path to ``ADCS/environment/de421.bsp``.
        :rtype: pathlib.Path

        """
        project_root = Path(__file__).resolve().parents[2]  # adjust if needed
        external_dir = project_root / "ADCS" / "environment"
        external_dir.mkdir(parents=True, exist_ok=True)
        return external_dir / "de421.bsp"

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
