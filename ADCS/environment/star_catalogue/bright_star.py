"""Offline Yale Bright Star Catalogue, fifth revised edition."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from .catalogue import NavigationStar, StarCatalog
from ADCS.helpers.asset_manager import AssetManager


class BrightStarCatalog(StarCatalog):
    """Load the bundled Bright Star Catalogue with the usual sky-query API.

    The original CDS V/50 ``catalog`` contains 9,110 entries. The bundled
    gzip file is about 0.55 MB. Entries without a J2000 position or visual
    magnitude are omitted. Positions are fixed at the catalogue's J2000 FK5
    epoch; no proper-motion or aberration correction is applied.

    Stars use Harvard Revised identifiers in ``hr_id`` and have ``hip_id=None``.

    Source: Hoffleit and Warren, 5th Revised Edition (1991), CDS catalogue
    V/50. See the project citation guide.
    """

    FILENAME = "bsc5_catalog.gz"
    SOURCE_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/V/50/catalog.gz"

    def __init__(
        self,
        max_vmag: float | None = 6.5,
        kernel_path: str | Path | None = None,
    ) -> None:
        r"""Select a magnitude limit and bundled catalogue path.

        :param max_vmag: Faintest visual magnitude to retain, or all if ``None``.
        :type max_vmag: float or None
        :param kernel_path: Optional path to a compatible compressed catalogue.
        :type kernel_path: str or pathlib.Path or None
        :return: None
        :rtype: None
        """
        self.max_vmag = max_vmag
        self.kernel_path = (
            Path(kernel_path).expanduser()
            if kernel_path is not None
            else AssetManager().get("bsc5")
        )
        self._stars: list[NavigationStar] | None = None

    @property
    def stars(self) -> list[NavigationStar]:
        r"""Load bright-star records from the bundled gzip file on first use.

        :return: Usable stars sorted from brightest to faintest.
        :rtype: list[NavigationStar]
        :raises FileNotFoundError: If the bundled catalogue is missing.
        :raises ValueError: If the catalogue has no usable stars.
        """
        if self._stars is None:
            self._stars = self._load_catalog(self.kernel_path)
        return self._stars

    def _load_catalog(self, path: Path) -> list[NavigationStar]:
        r"""Parse the CDS V/50 fixed-width J2000 coordinates and V magnitude.

        :param path: Compressed CDS catalogue path.
        :type path: pathlib.Path
        :return: Parsed navigation stars.
        :rtype: list[NavigationStar]
        :raises ValueError: If the source contains no usable entries.
        """
        stars = []
        with gzip.open(path, "rt", encoding="ascii") as source:
            for line in source:
                if len(line) < 107:
                    continue
                try:
                    hr_id = int(line[0:4])
                    ra_hour = int(line[75:77])
                    ra_minute = int(line[77:79])
                    ra_second = float(line[79:83])
                    dec_sign = line[83]
                    dec_degree = int(line[84:86])
                    dec_minute = int(line[86:88])
                    dec_second = int(line[88:90])
                    vmag = float(line[102:107])
                except ValueError:
                    continue  # Removed entries lack coordinates or magnitude.
                if (dec_sign not in "+-" or not 0 <= ra_hour < 24 or
                    not 0 <= ra_minute < 60 or not 0 <= ra_second < 60 or
                    not 0 <= dec_degree <= 90 or not 0 <= dec_minute < 60 or
                    not 0 <= dec_second < 60 or not np.isfinite(vmag)):
                    continue
                if self.max_vmag is not None and vmag > self.max_vmag:
                    continue
                ra = np.deg2rad(15.0 * (ra_hour + ra_minute / 60 + ra_second / 3600))
                dec_abs = dec_degree + dec_minute / 60 + dec_second / 3600
                dec = np.deg2rad(dec_abs if dec_sign == "+" else -dec_abs)
                direction = np.array((
                    np.cos(dec) * np.cos(ra),
                    np.cos(dec) * np.sin(ra),
                    np.sin(dec),
                ))
                name = line[4:14].strip() or f"HR {hr_id}"
                stars.append(NavigationStar(None, name, ra, dec, vmag, direction, hr_id))
        stars.sort(key=lambda star: star.vmag)
        if not stars:
            raise ValueError(f"No usable Bright Star Catalogue entries found in {path}")
        return stars


__all__ = ["BrightStarCatalog"]
