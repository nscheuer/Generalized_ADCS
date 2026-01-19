__all__ = ["Ephemeris"]

from pathlib import Path
from skyfield.api import load, Loader
from typing import Optional

class Ephemeris:
    r"""
    Provides access to high-precision planetary ephemerides using the 
    `Skyfield <https://rhodesmill.org/skyfield/>`_ library.

    This class manages loading (or downloading) the JPL DE421 ephemeris 
    file and initializes planetary objects such as the Earth and the Sun.
    It first attempts to load a local copy from 
    ``ADCS/environment/de421.bsp``. If the file or directory 
    does not exist, it is downloaded and stored there automatically.

    Parameters
    ----------
    filepath : Path, optional
        Path to a pre-downloaded ephemeris file. If ``None``, the loader 
        searches for ``ADCS/environment/de421.bsp`` relative to the project 
        root. If not found, it downloads the file to that location.

    Attributes
    ----------
    planets : skyfield.jpllib.SpiceKernel
        Loaded planetary ephemeris data.
    sun : skyfield.jpllib.Body
        The Sun body object from the loaded ephemeris.
    earth : skyfield.jpllib.Body
        The Earth body object from the loaded ephemeris.
    ts : skyfield.timelib.Timescale
        The Skyfield timescale object for creating ``Time`` instances.

    Notes
    -----
    The DE421 model covers the years 1900–2050 and provides accurate 
    positions and velocities of major solar system bodies in the 
    :math:`\text{ICRF}` reference frame.
    """

    def __init__(self, filepath: Optional[Path] = None) -> None:
        """Initialize the ephemeris loader and planetary objects."""
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

        # Create a timescale object (shared across all orbital computations)
        self.ts = load.timescale()

    # ----------------------------------------------------------------------
    def _get_default_ephemeris_path(self) -> Path:
        """
        Get the expected location for the local DE421 ephemeris file.

        Returns
        -------
        Path
            Path to ``ADCS/environment/de421.bsp`` relative to the project root.
        """
        project_root = Path(__file__).resolve().parents[2]  # adjust if needed
        external_dir = project_root / "ADCS" / "environment"
        external_dir.mkdir(parents=True, exist_ok=True)
        return external_dir / "de421.bsp"

    # ----------------------------------------------------------------------
    def _download_ephemeris(self, save_path: Path):
        """
        Download and store the DE421 ephemeris file to the given location.

        Parameters
        ----------
        save_path : Path
            Destination path where the ephemeris file will be saved.

        Returns
        -------
        planets : skyfield.jpllib.SpiceKernel
            The loaded Skyfield planetary ephemeris object.

        Raises
        ------
        RuntimeError
            If the ephemeris file could not be downloaded or loaded.
        """
        try:
            loader = Loader(str(save_path.parent))
            planets = loader("de421.bsp")  # downloads if missing
            print(f"✅ Ephemeris downloaded to: {planets.path}")
            return planets
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download DE421 ephemeris: {e}") from e
