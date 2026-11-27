"""Fast acceleration from Earth's zonal (J2 through J6) gravity terms.

Positions are Cartesian inertial metres. As for any axisymmetric zonal model,
the inertial Z axis is taken to be the body's symmetry axis; ``epoch_jd`` is
accepted for API compatibility and is not needed by this time-independent model.
"""

import numpy as np
from numba import njit

# Earth zonal harmonics, dimensionless, with the conventional positive J2 sign.
EARTH_ZONAL_COEFFICIENTS = np.array(
    (1.08262668e-3, -2.5324105e-6, -1.6198976e-6, -2.2729608e-7, 5.406812e-7),
    dtype=np.float64,
)
EARTH_MU_M3_S2 = 3.986004418e14
EARTH_REFERENCE_RADIUS_M = 6378137.0


@njit(cache=True)
def gravity_acceleration_inertial_m_s2(
    position_inertial_m: np.ndarray,
    epoch_jd: float,
    gravitational_parameter_m3_s2: float,
    reference_radius_m: float,
    zonal_coefficients: np.ndarray,
) -> np.ndarray:
    """Return central plus J2--J6 acceleration in inertial Cartesian axes.

    ``zonal_coefficients[k]`` is J(k+2); pass zeros to omit terms. ``epoch_jd``
    is intentionally unused because zonal gravity is invariant to rotation
    about its symmetry axis.
    """
    x, y, z = position_inertial_m[0], position_inertial_m[1], position_inertial_m[2]
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    if r == 0.0:
        raise ValueError("Gravity is undefined at the body's centre")

    inv_r = 1.0 / r
    s = z * inv_r
    ratio = reference_radius_m * inv_r
    ratio2 = ratio * ratio

    # Legendre polynomials P_n(s) and derivatives dP_n/ds, n=2,...,6.
    p = np.empty(5, dtype=np.float64)
    dp = np.empty(5, dtype=np.float64)
    p[0] = 0.5 * (3.0 * s * s - 1.0)
    dp[0] = 3.0 * s
    p[1] = 0.5 * (5.0 * s**3 - 3.0 * s)
    dp[1] = 0.5 * (15.0 * s * s - 3.0)
    p[2] = (35.0 * s**4 - 30.0 * s * s + 3.0) / 8.0
    dp[2] = (140.0 * s**3 - 60.0 * s) / 8.0
    p[3] = (63.0 * s**5 - 70.0 * s**3 + 15.0 * s) / 8.0
    dp[3] = (315.0 * s**4 - 210.0 * s * s + 15.0) / 8.0
    p[4] = (231.0 * s**6 - 315.0 * s**4 + 105.0 * s * s - 5.0) / 16.0
    dp[4] = (1386.0 * s**5 - 1260.0 * s**3 + 210.0 * s) / 16.0

    radial_factor = 1.0
    z_gradient_factor = 0.0
    ratio_n = ratio2
    count = min(len(zonal_coefficients), 5)
    for k in range(count):
        jn = zonal_coefficients[k]
        n = k + 2
        radial_factor -= (n + 1.0) * jn * ratio_n * p[k]
        z_gradient_factor += jn * ratio_n * dp[k]
        ratio_n *= ratio

    mu_over_r3 = gravitational_parameter_m3_s2 / (r2 * r)
    base = -mu_over_r3 * radial_factor
    # a = -grad of (-U): the latitude term enters with a negative sign,
    # -mu/r^3 * sum(Jn (R/r)^n Pn'(s)) * (r z_hat - s r_vec).
    result = np.empty(3, dtype=np.float64)
    result[0] = base * x - mu_over_r3 * z_gradient_factor * (-s * x)
    result[1] = base * y - mu_over_r3 * z_gradient_factor * (-s * y)
    result[2] = base * z - mu_over_r3 * z_gradient_factor * (r - s * z)
    return result


class JGravityModel:
    """Earth gravity model with a selectable subset of J2 through J6."""

    degree = None

    def __init__(
        self,
        degrees: tuple[int, ...] = (2, 3, 4, 5, 6),
        coefficients: np.ndarray | None = None,
    ) -> None:
        if self.degree is not None and degrees == (2, 3, 4, 5, 6):
            degrees = (self.degree,)
        self.degrees = tuple(sorted(set(degrees)))
        if any(degree not in (2, 3, 4, 5, 6) for degree in self.degrees):
            raise ValueError("J-gravity degrees must be selected from 2 through 6")
        source = EARTH_ZONAL_COEFFICIENTS if coefficients is None else coefficients
        if len(source) < 5:
            raise ValueError("Provide five coefficients ordered as J2, J3, J4, J5, J6")
        self.zonal_coefficients = np.zeros(5, dtype=np.float64)
        for degree in self.degrees:
            self.zonal_coefficients[degree - 2] = source[degree - 2]
        self.gravitational_parameter_m3_s2 = EARTH_MU_M3_S2
        self.reference_radius_m = EARTH_REFERENCE_RADIUS_M

    def evaluate(
        self, position_inertial_m: np.ndarray, epoch_jd: float = 0.0
    ) -> np.ndarray:
        return gravity_acceleration_inertial_m_s2(
            np.asarray(position_inertial_m, dtype=np.float64),
            epoch_jd,
            self.gravitational_parameter_m3_s2,
            self.reference_radius_m,
            self.zonal_coefficients,
        )

    acceleration = evaluate

    # Keep the low-level kernel available on the class as well as at module scope.
    acceleration_inertial_m_s2 = staticmethod(gravity_acceleration_inertial_m_s2)
