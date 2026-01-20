__all__ = ["NavigationStar", "StarCatalog"]

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from numpy.typing import NDArray

from ADCS.orbits.universal_constants import EarthConstants

@dataclass
class NavigationStar:
    r"""
    **Navigation Star Definition**

    This data class represents a single **navigation star** used by attitude
    sensors such as a star tracker.

    Each star is defined by its inertial pointing direction, brightness, and
    identifying metadata derived from astronomical catalogs (e.g. Hipparcos).

    ---
    **Coordinate Representation**

    The star direction is stored as a **unit vector in the Earth-Centered Inertial
    (ECI) frame**:

    .. math::

        \mathbf{s}_\mathrm{ECI}
        =
        \begin{bmatrix}
        \cos\delta\cos\alpha \\
        \cos\delta\sin\alpha \\
        \sin\delta
        \end{bmatrix}

    where:

    - :math:`\alpha` — right ascension (RA)
    - :math:`\delta` — declination (Dec)

    This representation assumes a distant (effectively infinite-range) star,
    so parallax effects are neglected.

    ---
    **Intended Use**

    Instances of this class are returned by
    :meth:`~ADCS.environment.StarCatalog.get_visible_stars`
    and consumed by attitude sensors such as
    :class:`~ADCS.satellite_hardware.sensors.star_tracker.StarTracker`.

    Attributes
    ----------
    hip_id : int
        Hipparcos catalog identifier.

    name : str
        Common star name.

    ra_rad : float
        Right ascension in radians.

    dec_rad : float
        Declination in radians.

    vmag : float
        Apparent visual magnitude.
        Lower values correspond to brighter stars.

    s_eci : ndarray, shape (3,)
        Unit vector pointing from the Earth toward the star,
        expressed in the ECI frame.

    References
    ----------
    Vallado, D. A., *Fundamentals of Astrodynamics and Applications*,
    4th ed., Section 2.3.
    """
    hip_id: int
    name: str
    ra_rad: float
    dec_rad: float
    vmag: float
    s_eci: NDArray[np.float64]


class StarCatalog:
    r"""
    **Navigation Star Catalog**

    This class provides a curated catalog of bright navigation stars and
    implements **geometric visibility checks** for star tracker sensors.

    The catalog is intentionally limited to relatively bright stars
    (visual magnitude :math:`V \lesssim 2.5`) to reflect realistic onboard
    star tracker operation and to keep visibility queries computationally
    lightweight.

    ---
    **Catalog Contents**

    Each star is represented by a :class:`~ADCS.environment.NavigationStar`
    object containing:

    - Right ascension and declination
    - Apparent visual magnitude
    - Precomputed inertial-frame unit direction

    The default catalog is derived from the **Hipparcos Catalog (ESA, 1997)**.

    ---
    **Visibility Model**

    A star is considered *visible* if and only if all of the following hold:

    1. The star lies within the sensor field of view (FOV)
    2. The star is not occluded by the Earth
    3. The star is not occluded by the Moon (if enabled)
    4. The sensor is not blinded by the Sun

    These checks are implemented using purely geometric criteria and are
    appropriate for simulation and estimator development.

    ---
    **Coordinate Frames**

    All directions are expressed in the **ECI frame**.
    Satellite position vectors are assumed to be given in kilometers.

    Attributes
    ----------
    R_EARTH : float
        Mean Earth radius [km].

    R_MOON : float
        Mean Moon radius [km].

    See Also
    --------
    ~ADCS.environment.NavigationStar  
    ~ADCS.satellite_hardware.sensors.star_tracker.StarTracker
    """
    R_EARTH: float = EarthConstants.R_e
    R_MOON: float = EarthConstants.R_moon

    def __init__(self) -> None:
        r"""
        Initialize the star catalog.

        The catalog is populated with a predefined list of bright navigation
        stars at construction time. Star directions are converted from
        right ascension and declination to inertial-frame unit vectors.

        Notes
        -----
        The catalog is static and does not model proper motion, parallax,
        or stellar aberration. These effects are negligible for typical
        small-satellite star tracker applications.
        """
        self._stars: List[NavigationStar] = self._init_catalog()

    def _init_catalog(self) -> List[NavigationStar]:
        r"""
        Construct the internal navigation star list.

        Star data are specified using right ascension and declination
        (in degrees for readability) and converted internally to radians
        and inertial-frame unit vectors.

        The conversion follows standard spherical-to-Cartesian mapping:

        .. math::

            \mathbf{s}_\mathrm{ECI}
            =
            \begin{bmatrix}
            \cos\delta\cos\alpha \\
            \cos\delta\sin\alpha \\
            \sin\delta
            \end{bmatrix}

        Returns
        -------
        list of ~ADCS.environment.NavigationStar
            List of initialized navigation star objects.

        References
        ----------
        Hipparcos Catalog, ESA (1997)  
        Vallado (2013), Section 2.3
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
        r"""
        Determine which navigation stars are visible to a star tracker.

        This method applies a sequence of geometric visibility checks
        to each catalog star.

        ---
        **Visibility Criteria**

        A star is considered visible if:

        1. **Field-of-view constraint**

           .. math::

               \arccos(\mathbf{b}^\top \mathbf{s}_\mathrm{ECI})
               \;\le\; \tfrac{1}{2}\,\mathrm{FOV}

        2. **Earth occlusion**

           The star is not located behind the Earth disk as seen from
           the spacecraft.

        3. **Moon occlusion (optional)**

           The star is not located behind the Moon disk.

        4. **Sun exclusion**

           If the Sun is closer than ``sun_exclusion_rad`` to the boresight,
           the tracker is considered **completely blinded** and no stars
           are returned.

        ---
        **Earth and Moon Occlusion**

        Occlusion checks are performed using angular radii:

        .. math::

            \theta_\oplus = \arcsin\!\left(\frac{R_\oplus}{\|\mathbf{r}_\mathrm{sat}\|}\right), \qquad
            \theta_\leftmoon = \arcsin\!\left(\frac{R_\leftmoon}{\|\mathbf{r}_{\leftmoon}\|}\right)

        Parameters
        ----------
        boresight_eci : ndarray, shape (3,)
            Star tracker boresight direction in the ECI frame (unit vector).

        fov_rad : float
            Full angular field of view of the star tracker [rad].

        r_sat_eci : ndarray, shape (3,)
            Spacecraft position in the ECI frame [km].

        sun_eci : ndarray, shape (3,), optional
            Sun position in the ECI frame [km].
            If provided, Sun exclusion checking is enabled.

        moon_eci : ndarray, shape (3,), optional
            Moon position in the ECI frame [km].
            If provided, Moon occlusion checking is enabled.

        sun_exclusion_rad : float, optional
            Minimum allowable Sun–boresight angular separation [rad].

        Returns
        -------
        list of ~ADCS.environment.NavigationStar
            List of visible navigation stars.
            Returns an empty list if the tracker is blinded by the Sun.

        References
        ----------
        Vallado (2013), Section 5.3 — Earth occlusion geometry  
        Liebe (2002), Section IV-A — Star tracker Sun exclusion
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
