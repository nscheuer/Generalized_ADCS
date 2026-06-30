from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

r"""
This module exposes the following constant singletons:

- PHYS
- EARTH
- TIME
- DEFAULT
- CG5

See their attributes via Python introspection.
"""

@dataclass(frozen=True)
class _PhysicalConstants:
    r"""
    Physical constants in SI or derived units.
    """
    c: float = 2.99792458e8          # m/s
    grav_const: float = 6.6742e-11   # m^3/kg/s^2
    solar_constant: float = 1361.0   # W/m^2


@dataclass(frozen=True)
class _EarthConstants:
    r"""
    Standard geophysical constants for Earth.

    The zonal harmonic coefficients ``J2coeff`` .. ``J6coeff`` are the
    *unnormalized* zonal coefficients :math:`J_n` defined through the geopotential

    .. math::

        U = \frac{\mu}{r}\left[1 - \sum_{n\ge 2} J_n
            \left(\frac{R_e}{r}\right)^n P_n(\sin\phi)\right],

    where :math:`\phi` is geocentric latitude and :math:`P_n` is the Legendre
    polynomial of degree ``n``. Values are the standard EGM96-derived constants
    (sign convention: :math:`J_2,J_6>0`, :math:`J_3,J_4,J_5<0`). They are
    collected in :attr:`Jcoeffs` (degrees 2..6) for the orbit propagator.
    """
    R_e: float = 6378.1363            # km
    R_moon: float = 1737.4            # km
    mu_e: float = 398600.4415         # km^3/s^2
    J2coeff: float = 1.082635854e-3
    J3coeff: float = -2.53265649e-6
    J4coeff: float = -1.61962159e-6
    J5coeff: float = -2.27296083e-7
    J6coeff: float = 5.40681239e-7
    J2: float = J2coeff * R_e**2 * mu_e  # km^5/s^2
    m_earth: float = 5.9736e24        # kg
    omega_e: float = 7.2921159e-5     # rad/s, Earth sidereal rotation rate (IERS)
    solar_constant: float = 1361.0    # W/m^2
    c: float = 299792458.0            # m/s

    @property
    def Jcoeffs(self) -> np.ndarray:
        r"""
        Unnormalized zonal harmonic coefficients for degrees 2 through 6.

        :return:
            Array ``[J2, J3, J4, J5, J6]``.
        :rtype: numpy.ndarray
        """
        return np.array(
            [self.J2coeff, self.J3coeff, self.J4coeff, self.J5coeff, self.J6coeff],
            dtype=float,
        )


@dataclass(frozen=True)
class _ThirdBodyConstants:
    r"""
    Gravitational parameters of the third bodies used for lunisolar
    (Sun and Moon) perturbations of Earth-orbiting spacecraft.
    """
    mu_sun: float = 1.32712440018e11   # km^3/s^2 (IAU/DE-consistent)
    mu_moon: float = 4902.800066       # km^3/s^2


@dataclass(frozen=True)
class _TimeConstants:
    r"""
    Time conversion factors and numerical tolerances.
    """
    cent2sec: float = 100.0 * 365.25 * 24.0 * 3600.0
    sec2cent: float = 1.0 / (100.0 * 365.25 * 24.0 * 3600.0)
    time_eps: float = 1.0e-3
    num_eps: float = 1.0e-16


@dataclass(frozen=True)
class _DefaultStates:
    r"""
    Default zero-state and small perturbation quantities.
    """
    zeroquat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    err_J_var: float = 0.0


@dataclass(frozen=True)
class _CG5Coefficients:
    r"""
    Coefficients for the 5-stage Commutator-Free Lie-Group (CG5) Integrator.
    """
    a: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8177227988124852, 0.0, 0.0, 0.0, 0.0],
        [0.3199876375476427, 0.0659864263556022, 0.0, 0.0, 0.0],
        [0.9214417194464946, 0.4997857776773573, -0.0969984448371582, 0.0, 0.0],
        [0.3552358559023322, 0.2390958372307326,
         0.3918565724203246, -0.1092979392113565, 0.0],
    ]))

    b: np.ndarray = field(default_factory=lambda: np.array([
        0.1370831520630755,
        -0.0183698531564020,
        0.7397813985370780,
        -0.1907142565505889,
        0.3322195591068374,
    ]))

    c: np.ndarray = field(default_factory=lambda: np.array([
        0.0,
        0.8177227988124852,
        0.3859740639032449,
        0.3242290522866937,
        0.8768903263420429,
    ]))


PhysicalConstants = _PhysicalConstants()
EarthConstants = _EarthConstants()
ThirdBodyConstants = _ThirdBodyConstants()
TimeConstants = _TimeConstants()
DefaultStates = _DefaultStates()
CG5 = _CG5Coefficients()

# backward-compatible aliases (runtime only)
R_e = EarthConstants.R_e
mu_e = EarthConstants.mu_e
J2 = EarthConstants.J2
c = PhysicalConstants.c
CG5_a = CG5.a
CG5_b = CG5.b
CG5_c = CG5.c


__all__ = [
    "PhysicalConstants",
    "EarthConstants",
    "ThirdBodyConstants",
    "TimeConstants",
    "DefaultStates",
    "CG5",
]
