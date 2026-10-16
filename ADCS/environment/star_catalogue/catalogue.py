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

    This class defines a single **navigation star** used by attitude
    determination sensors such as star trackers.

    A navigation star is modeled as a distant, fixed-direction celestial
    reference whose inertial pointing direction is known with high accuracy.
    The class encapsulates both the catalog metadata and the corresponding
    inertial-frame unit vector.

    Inertial Direction Model
    ------------------------

    Each star is assumed to be located at an effectively infinite distance.
    Therefore, its apparent direction is constant and independent of the
    spacecraft position. The star direction is represented as a **unit vector**
    in the Earth-Centered Inertial (ECI) frame.

    Given right ascension :math:`\alpha` and declination :math:`\delta`,
    the inertial direction vector is defined as:

    .. math::

        \mathbf{s}_{\mathrm{ECI}} =
        \begin{bmatrix}
            \cos\delta \cos\alpha \\
            \cos\delta \sin\alpha \\
            \sin\delta
        \end{bmatrix}

    This mapping follows the standard spherical-to-Cartesian conversion
    described in Vallado (2013).

    Intended Usage
    --------------

    Instances of this class are typically constructed by
    :class:`~ADCS.environment.StarCatalog` and returned by
    :meth:`~ADCS.environment.StarCatalog.get_visible_stars`.

    They are consumed by attitude sensors such as
    :class:`~ADCS.satellite_hardware.sensors.star_tracker.StarTracker`
    for centroiding, pattern matching, and attitude estimation.

    Assumptions
    -----------

    - Stellar parallax is neglected
    - Proper motion is neglected
    - Stellar aberration is neglected

    These approximations are appropriate for typical small-satellite
    star tracker simulations.

    :param hip_id: Hipparcos catalog identifier.
    :type hip_id: int

    :param name: Common or Bayer star name.
    :type name: str

    :param ra_rad: Right ascension in radians.
    :type ra_rad: float

    :param dec_rad: Declination in radians.
    :type dec_rad: float

    :param vmag: Apparent visual magnitude (lower is brighter).
    :type vmag: float

    :param s_eci: Unit direction vector toward the star in the ECI frame.
    :type s_eci: numpy.ndarray

    References
    ----------

    - Vallado, D. A., *Fundamentals of Astrodynamics and Applications*,
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
    implements **geometric visibility filtering** suitable for star tracker
    simulations and attitude determination algorithms.

    The catalog is intentionally limited to stars with relatively high
    apparent brightness to reflect realistic onboard sensor constraints
    and to reduce computational load.

    Catalog Composition
    -------------------

    Each catalog entry is represented by a
    :class:`~ADCS.environment.NavigationStar` object containing:

    - Hipparcos identifier
    - Common name
    - Right ascension and declination
    - Apparent visual magnitude
    - Precomputed ECI-frame unit direction

    The default catalog is derived from the **Hipparcos Catalog (ESA, 1997)**.

    Visibility Model Overview
    --------------------------

    A star is considered *visible* to a star tracker if all of the following
    geometric conditions are satisfied:

    1. The star lies within the tracker field of view (FOV)
    2. The star is not occluded by the Earth
    3. The star is not occluded by the Moon (optional)
    4. The tracker is not blinded by the Sun

    These checks are purely geometric and are independent of sensor noise,
    detector physics, or image processing effects.

    Coordinate Frames
    -----------------

    - All direction vectors are expressed in the Earth-Centered Inertial (ECI)
      frame.
    - Position vectors are expressed in kilometers.
    - Angular quantities are expressed in radians.


    :param R_EARTH: Mean Earth radius [km].
    :type R_EARTH: float

    :param R_MOON: Mean Moon radius [km].
    :type R_MOON: float

    See Also
    --------

    - :class:`~ADCS.environment.NavigationStar`
    - :class:`~ADCS.satellite_hardware.sensors.star_tracker.StarTracker`

    """
    R_EARTH: float = EarthConstants.R_e
    R_MOON: float = EarthConstants.R_moon

    def __init__(self) -> None:
        r"""
        Initialize the navigation star catalog.

        This constructor populates the internal catalog with a predefined list
        of bright navigation stars. Each star is defined by right ascension,
        declination, and apparent visual magnitude.

        During initialization:

        - Angular coordinates are converted from degrees to radians
        - Inertial-frame unit vectors are computed and stored
        - The catalog is cached for repeated visibility queries

        Model Limitations
        -----------------

        The catalog is static and does not account for:

        - Stellar proper motion
        - Annual parallax
        - Relativistic aberration

        These effects are negligible for most Earth-orbiting spacecraft star
        tracker applications.

        :return: None
        :rtype: None

        """
        self._stars: List[NavigationStar] = self._init_catalog()

    def _init_catalog(self) -> List[NavigationStar]:
        r"""
        Construct the internal navigation star list.

        This method defines the catalog contents using a hard-coded list of
        bright stars specified by right ascension, declination, and visual
        magnitude.

        Angular coordinates are provided in degrees for readability and are
        internally converted to radians.

        Direction Vector Conversion
        ---------------------------

        For each star, the inertial direction vector is computed using:

        .. math::

            \mathbf{s}_{\mathrm{ECI}} =
            \begin{bmatrix}
                \cos\delta \cos\alpha \\
                \cos\delta \sin\alpha \\
                \sin\delta
            \end{bmatrix}

        where:

        - :math:`\alpha` is the right ascension
        - :math:`\delta` is the declination

        The resulting vector is normalized by construction.

        :return: List of initialized navigation star objects.
        :rtype: list[~ADCS.environment.NavigationStar]

        References
        ----------

        - Hipparcos Catalog, ESA (1997)
        - Vallado (2013), Section 2.3

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
        r"""
        Return the list of navigation stars in the catalog.

        This property provides read-only access to the internally stored list
        of :class:`~ADCS.environment.NavigationStar` objects.

        :return: List of navigation stars.
        :rtype: list[~ADCS.environment.NavigationStar]

        """
        return self._stars

    def _catalog_arrays(self):
        r"""Cache numeric star directions and sky coordinates for queries.

        :return: Stars, ECI unit vectors, right ascensions, and declinations.
        :rtype: tuple
        """
        stars = self.stars
        if getattr(self, "_vectors", None) is None:
            self._vectors = np.stack([star.s_eci for star in stars]) if stars else np.empty((0, 3))
            self._ra = np.fromiter((star.ra_rad for star in stars), dtype=float, count=len(stars))
            self._dec = np.fromiter((star.dec_rad for star in stars), dtype=float, count=len(stars))
        return stars, self._vectors, self._ra, self._dec

    def stars_in_cone(
        self,
        boresight_eci: NDArray[np.float64],
        radius_rad: float,
        max_vmag: float | None = None,
    ) -> List[NavigationStar]:
        """Return stars within an angular radius of a unit ECI direction.

        Results retain catalogue order, which is brightest first. This is a
        sky query only; use ``get_visible_stars`` for spacecraft occlusion.

        :param boresight_eci: Center direction in the inertial frame.
        :type boresight_eci: numpy.ndarray
        :param radius_rad: Angular search radius [rad].
        :type radius_rad: float
        :param max_vmag: Optional faintest allowed visual magnitude.
        :type max_vmag: float or None
        :return: Catalogue stars inside the cone, brightest first.
        :rtype: list[NavigationStar]
        :raises ValueError: If radius or direction is invalid.
        """
        if not 0.0 <= radius_rad <= np.pi:
            raise ValueError("radius_rad must be between 0 and pi")
        direction = np.asarray(boresight_eci, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("boresight_eci must be a finite 3-vector")
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            raise ValueError("boresight_eci must be nonzero")
        direction = direction / norm
        stars, vectors, _, _ = self._catalog_arrays()
        if not stars:
            return []
        inside = vectors @ direction >= np.cos(radius_rad) - 1e-14
        return [star for star, selected in zip(stars, inside) if selected and (max_vmag is None or star.vmag <= max_vmag)]

    def stars_in_region(
        self,
        ra_min_rad: float,
        ra_max_rad: float,
        dec_min_rad: float,
        dec_max_rad: float,
        max_vmag: float | None = None,
    ) -> List[NavigationStar]:
        """Load a rectangular sky swath in ICRS right ascension/declination.

        Right ascension wraps through zero when ``ra_min_rad > ra_max_rad``.
        Angles are radians; results retain brightness order.

        :param ra_min_rad: Lower right ascension boundary [rad].
        :type ra_min_rad: float
        :param ra_max_rad: Upper right ascension boundary [rad].
        :type ra_max_rad: float
        :param dec_min_rad: Lower declination boundary [rad].
        :type dec_min_rad: float
        :param dec_max_rad: Upper declination boundary [rad].
        :type dec_max_rad: float
        :param max_vmag: Optional faintest allowed visual magnitude.
        :type max_vmag: float or None
        :return: Catalogue stars inside the sky region, brightest first.
        :rtype: list[NavigationStar]
        :raises ValueError: If angles or declination bounds are invalid.
        """
        values = (ra_min_rad, ra_max_rad, dec_min_rad, dec_max_rad)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("sky-region angles must be finite")
        if not -np.pi / 2 <= dec_min_rad <= dec_max_rad <= np.pi / 2:
            raise ValueError("declination bounds must be ordered and within [-pi/2, pi/2]")
        full_ra = abs(ra_max_rad - ra_min_rad) >= 2 * np.pi
        ra_min = ra_min_rad % (2 * np.pi)
        ra_max = ra_max_rad % (2 * np.pi)
        stars, _, ra, dec = self._catalog_arrays()
        ra = ra % (2 * np.pi)
        if full_ra:
            ra_mask = np.ones(len(stars), dtype=bool)
        elif ra_min <= ra_max:
            ra_mask = (ra >= ra_min) & (ra <= ra_max)
        else:
            ra_mask = (ra >= ra_min) | (ra <= ra_max)
        mask = ra_mask & (dec >= dec_min_rad) & (dec <= dec_max_rad)
        if max_vmag is not None:
            mask &= np.fromiter((star.vmag <= max_vmag for star in stars), dtype=bool, count=len(stars))
        return [star for star, selected in zip(stars, mask) if selected]

    load_sky_swath = stars_in_region

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

        This method applies a sequence of geometric visibility checks to every
        star in the catalog and returns only those that satisfy all criteria.

        1. Field-of-View Constraint
        ---------------------------

        A star is inside the sensor field of view if the angular separation
        between the boresight direction :math:`\mathbf{b}` and the star direction
        :math:`\mathbf{s}` satisfies:

        .. math::

            \arccos(\mathbf{b}^\top \mathbf{s})
            \le \frac{\mathrm{FOV}}{2}

        2. Earth Occlusion
        ------------------

        The Earth subtends an angular radius as seen from the spacecraft:

        .. math::

            \theta_\oplus = \arcsin\left(\frac{R_\oplus}{\|\mathbf{r}_{\mathrm{sat}}\|}\right)

        Let :math:`\mathbf{n} = -\mathbf{r}_{\mathrm{sat}} / \|\mathbf{r}_{\mathrm{sat}}\|`
        be the nadir direction. A star is considered **occluded by Earth** if:

        .. math::

            \arccos(\mathbf{n}^\top \mathbf{s}) < \theta_\oplus

        3. Moon Occlusion (Optional)
        ----------------------------

        If the Moon position is provided, its angular radius is computed as:

        .. math::

            \theta_{\leftmoon} =
            \arcsin\left(\frac{R_{\leftmoon}}{\|\mathbf{r}_{\leftmoon} - \mathbf{r}_{\mathrm{sat}}\|}\right)

        The star is rejected if it lies within this angular radius.

        4. Sun Exclusion
        -----------------

        If the Sun direction lies closer than ``sun_exclusion_rad`` to the
        boresight, the star tracker is assumed to be completely blinded and
        **no stars are returned**.

        .. math::

            \arccos(\mathbf{b}^\top \mathbf{s}_{\odot}) < \theta_{\mathrm{excl}}

        :param boresight_eci: Star tracker boresight direction (unit vector).
        :type boresight_eci: numpy.ndarray

        :param fov_rad: Full angular field of view of the star tracker [rad].
        :type fov_rad: float

        :param r_sat_eci: Spacecraft position in the ECI frame [km].
        :type r_sat_eci: numpy.ndarray

        :param sun_eci: Sun position in the ECI frame [km].
        :type sun_eci: numpy.ndarray | None

        :param moon_eci: Moon position in the ECI frame [km].
        :type moon_eci: numpy.ndarray | None

        :param sun_exclusion_rad: Minimum allowable Sun–boresight separation [rad].
        :type sun_exclusion_rad: float

        :return: List of visible navigation stars.
        :rtype: list[~ADCS.environment.NavigationStar]

        References
        ----------

        - Vallado (2013), Section 5.3 — Earth occlusion geometry
        - Liebe (2002), Section IV-A — Star tracker Sun exclusion

        """
        half_fov = fov_rad / 2.0
        stars, vectors, _, _ = self._catalog_arrays()
        if not stars:
            return []
        boresight = np.asarray(boresight_eci, dtype=float)
        boresight_norm = np.linalg.norm(boresight)
        if boresight_norm == 0.0:
            raise ValueError("boresight_eci must be nonzero")
        boresight = boresight / boresight_norm
        visible_mask = vectors @ boresight >= np.cos(half_fov)

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
            if np.dot(boresight, sun_dir) > np.cos(sun_exclusion_rad):
                return []  # Tracker completely blinded - no stars visible

        # 2. Earth occlusion (star behind Earth).
        visible_mask &= vectors @ nadir <= np.cos(earth_angular_radius)

        # 3. Moon occlusion (star behind Moon).
        if moon_dir is not None:
            visible_mask &= vectors @ moon_dir <= np.cos(moon_angular_radius)

        return [star for star, selected in zip(stars, visible_mask) if selected]
