"""
Star catalog for navigation star tracker simulations.

This module provides a catalog of bright navigation stars from the Hipparcos
catalog, suitable for star tracker sensor simulation. Star positions are
given in the J2000 ECI (Earth-Centered Inertial) reference frame.

Mathematical Model:
    Given Right Ascension (α) and Declination (δ) in radians, the ECI
    unit vector is:

        s_ECI = [cos(δ)cos(α), cos(δ)sin(α), sin(δ)]^T

References:
    [1] ESA, "The Hipparcos and Tycho Catalogues", ESA SP-1200 (1997)
        Section 1.2: Celestial coordinate systems
    [2] Liebe, C.C., "Star Trackers for Attitude Determination",
        IEEE Aerospace and Electronic Systems Magazine (1995)
    [3] Vallado, D.A., "Fundamentals of Astrodynamics and Applications",
        4th Ed., Microcosm Press (2013), Section 5.3
"""
from __future__ import annotations

__all__ = ["NavigationStar", "StarCatalog"]

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from numpy.typing import NDArray


@dataclass
class NavigationStar:
    """A navigation star with catalog data.

    Attributes:
        hip_id: Hipparcos catalog ID
        name: Common name (may be empty)
        ra_rad: Right ascension in radians (J2000)
        dec_rad: Declination in radians (J2000)
        vmag: Visual magnitude
        s_eci: Unit vector in J2000 ECI frame, shape (3,)
    """
    hip_id: int
    name: str
    ra_rad: float
    dec_rad: float
    vmag: float
    s_eci: NDArray[np.float64]


class StarCatalog:
    """Catalog of bright navigation stars for star tracker simulation.

    Contains ~30 bright stars (Vmag < 2.5) suitable for navigation.
    Star positions are in J2000 ECI frame.

    The catalog includes the brightest stars visible from Earth, covering
    both northern and southern hemispheres. These stars are commonly used
    as navigation references in star tracker systems.

    References:
        [1] Hipparcos Catalog (ESA, 1997)

    Example:
        >>> catalog = StarCatalog()
        >>> print(f"Catalog has {len(catalog.stars)} stars")
        >>> sirius = catalog.stars[0]  # Brightest star
        >>> print(f"Sirius: RA={np.rad2deg(sirius.ra_rad):.1f}°")
    """

    # Earth and Moon radii for occlusion calculations (km)
    R_EARTH: float = 6378.137
    R_MOON: float = 1737.4

    def __init__(self) -> None:
        """Initialize the star catalog with bright navigation stars."""
        self._stars: List[NavigationStar] = self._init_catalog()

    def _init_catalog(self) -> List[NavigationStar]:
        """Initialize catalog with bright navigation stars.

        Star data format: (HIP_ID, name, RA_deg, Dec_deg, Vmag)
        RA/Dec are given in degrees for readability, then converted to radians.

        Reference: Hipparcos Catalog (ESA, 1997)
        """
        catalog_data = [
            # Brightest stars (Vmag < 1.0)
            (32349, "Sirius", 101.287, -16.716, -1.46),
            (30438, "Canopus", 95.988, -52.696, -0.72),
            (71683, "Alpha Centauri", 219.902, -60.834, -0.27),
            (69673, "Arcturus", 213.915, 19.182, -0.05),
            (91262, "Vega", 279.235, 38.784, 0.03),
            (24436, "Capella", 79.172, 45.998, 0.08),
            (24608, "Rigel", 78.634, -8.202, 0.13),
            (37279, "Procyon", 114.826, 5.225, 0.34),
            (27989, "Betelgeuse", 88.793, 7.407, 0.42),
            (7588, "Achernar", 24.429, -57.237, 0.46),
            (68702, "Hadar", 210.956, -60.373, 0.61),
            (97649, "Altair", 297.696, 8.868, 0.76),
            (21421, "Aldebaran", 68.980, 16.509, 0.85),
            (65474, "Spica", 201.298, -11.161, 0.97),
            # Bright stars (1.0 <= Vmag < 2.0)
            (80763, "Antares", 247.352, -26.432, 1.09),
            (37826, "Pollux", 116.329, 28.026, 1.14),
            (113368, "Fomalhaut", 344.413, -29.622, 1.16),
            (49669, "Deneb", 310.358, 45.280, 1.25),
            (62434, "Mimosa", 191.930, -59.689, 1.25),
            (60718, "Acrux", 186.650, -63.099, 1.33),
            (25336, "Bellatrix", 81.283, 6.350, 1.64),
            (25930, "Alnilam", 84.053, -1.202, 1.69),
            (26311, "Alnitak", 85.190, -1.943, 1.77),
            (9884, "Mirfak", 51.081, 49.861, 1.79),
            # Navigation stars (2.0 <= Vmag < 2.5)
            (11767, "Polaris", 37.954, 89.264, 2.02),
            (5447, "Mirach", 17.433, 35.621, 2.05),
            (677, "Alpheratz", 2.097, 29.091, 2.06),
            (28360, "Saiph", 86.939, -9.670, 2.07),
            (3179, "Schedar", 10.127, 56.537, 2.23),
            (746, "Caph", 2.295, 59.150, 2.27),
        ]

        stars = []
        deg2rad = np.pi / 180.0

        for hip_id, name, ra_deg, dec_deg, vmag in catalog_data:
            ra = ra_deg * deg2rad
            dec = dec_deg * deg2rad
            # Convert RA/Dec to ECI unit vector
            # Reference: Vallado (2013), coordinate transformations
            s_eci = np.array([
                np.cos(dec) * np.cos(ra),
                np.cos(dec) * np.sin(ra),
                np.sin(dec)
            ], dtype=np.float64)
            stars.append(NavigationStar(hip_id, name, ra, dec, vmag, s_eci))

        return stars

    @property
    def stars(self) -> List[NavigationStar]:
        """Get the list of navigation stars."""
        return self._stars

    def get_visible_stars(
        self,
        boresight_eci: NDArray[np.float64],
        fov_rad: float,
        r_sat_eci: NDArray[np.float64],
        sun_eci: Optional[NDArray[np.float64]] = None,
        moon_eci: Optional[NDArray[np.float64]] = None,
        sun_exclusion_rad: float = np.deg2rad(25.0)
    ) -> List[NavigationStar]:
        """Get stars visible in the field of view, accounting for occlusion.

        This method implements full visibility checking including:
        1. Field of view constraints
        2. Earth occlusion (star behind Earth disk)
        3. Moon occlusion (star behind Moon disk)
        4. Sun exclusion (tracker blinded by stray light)

        Args:
            boresight_eci: Star tracker boresight direction in ECI frame
                (unit vector, shape (3,))
            fov_rad: Full field of view in radians
            r_sat_eci: Satellite position in ECI frame (km), shape (3,)
            sun_eci: Sun position in ECI frame (km), optional.
                If provided, enables sun exclusion checking.
            moon_eci: Moon position in ECI frame (km), optional.
                If provided, enables Moon occlusion checking.
            sun_exclusion_rad: Sun exclusion angle from boresight in radians.
                Tracker is blinded if sun is closer than this angle.
                Typical values: 25-45 degrees. Default: 25 degrees.

        Returns:
            List of NavigationStar objects that are visible.
            Returns empty list if tracker is blinded by sun.

        References:
            [1] Vallado (2013), Section 5.3 for occlusion geometry
            [2] Liebe (2002), Section IV-A for sun exclusion angles

        Example:
            >>> catalog = StarCatalog()
            >>> r_sat = np.array([6778.0, 0.0, 0.0])  # 400 km altitude
            >>> boresight = np.array([0.0, 1.0, 0.0])  # Looking +Y
            >>> visible = catalog.get_visible_stars(boresight, np.deg2rad(20), r_sat)
        """
        visible = []
        half_fov = fov_rad / 2.0

        # Precompute Earth occlusion geometry
        # Reference: Vallado (2013), Section 5.3
        r_sat_norm = np.linalg.norm(r_sat_eci)
        earth_angular_radius = np.arcsin(self.R_EARTH / r_sat_norm)
        nadir = -r_sat_eci / r_sat_norm

        # Precompute Moon geometry if available
        if moon_eci is not None:
            r_to_moon = moon_eci - r_sat_eci
            moon_dist = np.linalg.norm(r_to_moon)
            moon_angular_radius = np.arcsin(self.R_MOON / moon_dist)
            moon_dir = r_to_moon / moon_dist
        else:
            moon_angular_radius = 0.0
            moon_dir = None

        # Check sun exclusion (tracker completely blinded if sun too close)
        # Reference: Liebe (2002), Section IV-A
        if sun_eci is not None:
            sun_dir = sun_eci / np.linalg.norm(sun_eci)
            sun_angle_from_boresight = np.arccos(np.clip(
                np.dot(boresight_eci, sun_dir), -1.0, 1.0
            ))
            if sun_angle_from_boresight < sun_exclusion_rad:
                return []  # Tracker completely blinded - no stars visible

        for star in self._stars:
            # 1. Check if star is within FOV
            cos_angle = np.dot(boresight_eci, star.s_eci)
            if cos_angle < np.cos(half_fov):
                continue

            # 2. Check Earth occlusion (star behind Earth)
            angle_from_nadir = np.arccos(np.clip(
                np.dot(nadir, star.s_eci), -1.0, 1.0
            ))
            if angle_from_nadir < earth_angular_radius:
                continue  # Star is behind Earth

            # 3. Check Moon occlusion (star behind Moon)
            if moon_dir is not None:
                angle_from_moon = np.arccos(np.clip(
                    np.dot(moon_dir, star.s_eci), -1.0, 1.0
                ))
                if angle_from_moon < moon_angular_radius:
                    continue  # Star is behind Moon

            visible.append(star)

        return visible
