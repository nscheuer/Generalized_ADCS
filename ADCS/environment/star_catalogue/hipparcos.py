"""Hipparcos main catalogue from the Centre de Donnees astronomiques de Strasbourg."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ADCS.environment.ephemeris._download import download_with_progress, kernel_path

from .catalogue import NavigationStar, StarCatalog


class HipparcosCatalog(StarCatalog):
    """Full Hipparcos catalogue, downloaded on first query and cached on disk.

    ``max_vmag`` limits stars loaded into memory; ``None`` loads all entries
    with a valid visual magnitude and position. Positions are fixed at the
    catalogue epoch J1991.25 in ICRS, with no proper-motion correction.
    """

    URL = "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat"
    FILENAME = "hip_main.dat"

    def __init__(self, max_vmag: float | None = 6.0, kernel_dir: str | Path | None = None) -> None:
        r"""Configure the catalogue magnitude limit and file location.

        :param max_vmag: Faintest visual magnitude to retain, or all if ``None``.
        :type max_vmag: float or None
        :param kernel_dir: Directory for the Hipparcos source file.
        :type kernel_dir: str or pathlib.Path or None
        :return: None
        :rtype: None
        """
        self.max_vmag = max_vmag
        self.kernel_dir = Path(kernel_dir) if kernel_dir is not None else kernel_path(self.FILENAME).parent
        self._stars: list[NavigationStar] | None = None

    @property
    def stars(self) -> list[NavigationStar]:
        r"""Load and cache Hipparcos navigation stars on first access.

        :return: Stars with valid positions and magnitudes, brightest first.
        :rtype: list[NavigationStar]
        :raises ValueError: If the source file contains no usable stars.
        """
        if self._stars is None:
            path = self.kernel_dir / self.FILENAME
            if not path.is_file():
                self.kernel_dir.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + ".part")
                try:
                    download_with_progress(self.URL, partial, "Hipparcos catalogue")
                    stars = self._load_catalog(partial)
                    partial.replace(path)
                    self._stars = stars
                finally:
                    partial.unlink(missing_ok=True)
            else:
                self._stars = self._load_catalog(path)
        return self._stars

    def _load_catalog(self, path: Path) -> list[NavigationStar]:
        r"""Read the CDS fixed-width Hipparcos main catalogue.

        :param path: Local catalogue source file.
        :type path: pathlib.Path
        :return: Parsed stars subject to the magnitude limit.
        :rtype: list[NavigationStar]
        :raises ValueError: If no usable entries were parsed.
        """
        stars = []
        with path.open("r", encoding="ascii") as source:
            for line in source:
                # CDS I/239 ReadMe: HIP 9-14, Vmag 42-46, RA 52-63,
                # DE 65-76 (one-based inclusive byte positions).
                if len(line) < 76:
                    continue
                try:
                    hip_id = int(line[8:14])
                    vmag = float(line[41:46])
                    ra_deg = float(line[51:63])
                    dec_deg = float(line[64:76])
                except ValueError:
                    continue  # Some source entries have no astrometric solution.
                if not np.isfinite(vmag) or not 0 <= ra_deg <= 360 or not -90 <= dec_deg <= 90:
                    continue
                if self.max_vmag is not None and vmag > self.max_vmag:
                    continue
                ra = np.deg2rad(ra_deg)
                dec = np.deg2rad(dec_deg)
                direction = np.array((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))
                stars.append(NavigationStar(hip_id, f"HIP {hip_id}", ra, dec, vmag, direction))
        stars.sort(key=lambda star: star.vmag)
        if not stars:
            raise ValueError(f"No usable Hipparcos stars found in {path}")
        return stars

