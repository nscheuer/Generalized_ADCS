"""JPL DE442 planetary ephemeris, downloaded from the NAIF kernel archive."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from skyfield.api import load

from .de421 import DE421
from ADCS.helpers.asset_manager import AssetManager


class DE442(DE421):
    """DE442 implementation with the same interface and lifecycle as DE421.

    The 114 MB kernel is fetched from JPL/NAIF on first use and stored in
    ``ADCS/environment/kernels``. ``ADCS_DE442_EPHEMERIS_PATH`` can point to a custom
    file, or ``filepath`` can be supplied to the constructor.
    """

    DOWNLOAD_URL = (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        "spk/planets/de442.bsp"
    )
    KERNEL_FILENAME = "de442.bsp"
    ENV_PATH_VARIABLE = "ADCS_DE442_EPHEMERIS_PATH"

    @classmethod
    def _packaged_ephemeris_path(cls) -> Path:
        r"""Locate an optional source-package DE442 kernel.

        :return: Path within ``environment/kernels``.
        :rtype: pathlib.Path
        """
        return Path(__file__).resolve().parents[1] / "kernels" / cls.KERNEL_FILENAME

    @classmethod
    def _cache_ephemeris_path(cls) -> Path:
        r"""Choose the DE442 cache path for the current platform.

        :return: User override or platform cache destination.
        :rtype: pathlib.Path
        """
        override = os.environ.get(cls.ENV_PATH_VARIABLE)
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
        return base / "generalized_adcs" / cls.KERNEL_FILENAME

    def _download_ephemeris(self, save_path: Path):
        r"""Download DE442 from JPL NAIF and open it with Skyfield.

        :param save_path: Destination for ``de442.bsp``.
        :type save_path: pathlib.Path
        :return: Loaded planetary ephemeris kernel.
        :rtype: skyfield.jpllib.SpiceKernel
        :raises RuntimeError: If download or initialization fails.
        """
        save_path = Path(save_path)
        name = self.KERNEL_FILENAME.removesuffix(".bsp")
        return load(str(AssetManager().get(name, destination=save_path)))


__all__ = ["DE442"]
