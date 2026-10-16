"""JPL DE442 planetary ephemeris, downloaded from the NAIF kernel archive."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from skyfield.api import load

from .de421 import DE421
from ._download import download_with_progress


class DE442(DE421):
    """DE442 implementation with the same interface and lifecycle as DE421.

    The 114 MB kernel is fetched from JPL/NAIF on first use and stored in the
    user's cache. ``ADCS_DE442_EPHEMERIS_PATH`` can point to a custom cached
    file, or ``filepath`` can be supplied to the constructor.
    """

    DOWNLOAD_URL = (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        "spk/planets/de442.bsp"
    )
    KERNEL_FILENAME = "de442.bsp"
    ENV_PATH_VARIABLE = "ADCS_DE442_EPHEMERIS_PATH"

    @staticmethod
    def _packaged_ephemeris_path() -> Path:
        r"""Locate an optional source-package DE442 kernel.

        :return: Path within ``environment/kernels``.
        :rtype: pathlib.Path
        """
        return Path(__file__).resolve().parents[1] / "kernels" / "de442.bsp"

    @staticmethod
    def _cache_ephemeris_path() -> Path:
        r"""Choose the DE442 cache path for the current platform.

        :return: User override or platform cache destination.
        :rtype: pathlib.Path
        """
        override = os.environ.get("ADCS_DE442_EPHEMERIS_PATH")
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
        return base / "generalized_adcs" / "de442.bsp"

    def _download_ephemeris(self, save_path: Path):
        r"""Download DE442 from JPL NAIF and open it with Skyfield.

        :param save_path: Destination for ``de442.bsp``.
        :type save_path: pathlib.Path
        :return: Loaded planetary ephemeris kernel.
        :rtype: skyfield.jpllib.SpiceKernel
        :raises RuntimeError: If download or initialization fails.
        """
        save_path = Path(save_path)
        temporary_path = save_path.with_name(save_path.name + ".part")
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            download_with_progress(self.DOWNLOAD_URL, temporary_path, self.KERNEL_FILENAME)
            temporary_path.replace(save_path)
            planets = load(str(save_path))
            print(f"✅ DE442 ephemeris downloaded to: {save_path}")
            return planets
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"❌ Failed to download DE442 ephemeris: {exc}") from exc


__all__ = ["DE442"]
