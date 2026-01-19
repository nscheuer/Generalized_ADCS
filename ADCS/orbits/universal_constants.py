r"""
Global, immutable constants used throughout the spacecraft dynamics and control codebase.

This module provides:

- **Physical constants** (e.g., gravitational constant, speed of light)
- **Planetary constants** (Earth radius, mass, J2 term)
- **Numerical tolerances** (machine epsilon, time epsilon)
- **Integration coefficients** (for CG5 Lie-group integrator)

All constants are grouped into dataclasses for clarity and immutability.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ===============================================================
# 1. PHYSICAL CONSTANTS
# ===============================================================

@dataclass(frozen=True)
class PhysicalConstants:
    r"""
    Physical constants in SI or derived units.

    .. math::

        G &= 6.6742\times10^{-11}\ \text{m}^3\text{kg}^{-1}\text{s}^{-2} \\
        c &= 2.99792458\times10^{8}\ \text{m/s}

    
    Attributes
    ----------
    c : float
        Speed of light :math:`[m/s]`
    grav_const : float
        Universal gravitational constant :math:`G\ [m^3/kg/s^2]`
    solar_constant : float
        Solar irradiance constant :math:`S_0\ [W/m^2]`
    """
    c: float = 2.99792458e8  # m/s
    grav_const: float = 6.6742e-11  # m^3/kg/s^2
    solar_constant: float = 1361.0  # W/m^2


# ===============================================================
# 2. EARTH / PLANETARY CONSTANTS
# ===============================================================

@dataclass(frozen=True)
class EarthConstants:
    r"""
    Standard geophysical constants for Earth.

    .. math::

        R_E &= 6378.1363\ \text{km} \\
        \mu_E &= 398600.4415\ \text{km}^3/\text{s}^2 \\
        J_2 &= 1.082635854\times10^{-3}

    
    Attributes
    ----------
    R_e : float
        Mean equatorial radius [km]
    mu_e : float
        Standard gravitational parameter [km³/s²]
    J2coeff : float
        Dimensionless J2 zonal harmonic coefficient [-]
    J2 : float
        Precomputed :math:`J_2 R_E^2 \mu_E` term [km⁵/s²]
    m_earth : float
        Mass of Earth [kg]
    """
    R_e: float = 6378.1363  # km
    R_moon: float = 1737.4  # km
    mu_e: float = 398600.4415  # km³/s²
    J2coeff: float = 1.082635854e-3
    J2: float = J2coeff * R_e**2 * mu_e  # km⁵/s²
    m_earth: float = 5.9736e24  # kg
    solar_constant: float = 1361.0 # W/m^2
    c: float = 299792458.0 #speed of light, m/s


# ===============================================================
# 3. TIME AND NUMERICAL CONSTANTS
# ===============================================================

@dataclass(frozen=True)
class TimeConstants:
    r"""
    Time conversion factors and numerical tolerances.

    .. math::

        1\ \text{century} &= 100\ \text{years} = 3.15576\times10^9\ \text{s}

    
    Attributes
    ----------
    cent2sec : float
        Seconds per Julian century [s/century]
    sec2cent : float
        Inverse of ``cent2sec``
    time_eps : float
        Time tolerance used in integration comparisons
    num_eps : float
        Numerical tolerance / machine epsilon
    """
    cent2sec: float = 100.0 * 365.25 * 24.0 * 3600.0
    sec2cent: float = 1.0 / (100.0 * 365.25 * 24.0 * 3600.0)
    time_eps: float = 1.0e-3
    num_eps: float = 1.0e-16


# ===============================================================
# 4. DEFAULT QUATERNION AND ERRORS
# ===============================================================

@dataclass(frozen=True)
class DefaultStates:
    r"""
    Default zero-state and small perturbation quantities.

    
    Attributes
    ----------
    zeroquat : numpy.ndarray
        Default quaternion (no rotation): :math:`[1,0,0,0]`
    err_J_var : float
        Default error variance for inertia matrix terms
    """
    zeroquat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    err_J_var: float = 0.0


# ===============================================================
# 5. CG5 INTEGRATION COEFFICIENTS
# ===============================================================

@dataclass(frozen=True)
class CG5Coefficients:
    r"""
    Coefficients for the 5-stage Commutator-Free Lie-Group (CG5) Integrator.

    
    Attributes
    ----------
    a : numpy.ndarray
        Coefficient matrix for intermediate stages.
    b : numpy.ndarray
        Final combination weights.
    c : numpy.ndarray
        Time nodes (fractional step positions).
    """

    a: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8177227988124852, 0.0, 0.0, 0.0, 0.0],
        [0.3199876375476427, 0.0659864263556022, 0.0, 0.0, 0.0],
        [0.9214417194464946, 0.4997857776773573, -0.0969984448371582, 0.0, 0.0],
        [0.3552358559023322, 0.2390958372307326, 0.3918565724203246, -0.1092979392113565, 0.0],
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


# ===============================================================
# 6. EXPORT INSTANCES (READ-ONLY SINGLETONS)
# ===============================================================

PHYS = PhysicalConstants()
EARTH = EarthConstants()
TIME = TimeConstants()
DEFAULT = DefaultStates()
CG5 = CG5Coefficients()

# convenient aliases for backward compatibility
R_e = EARTH.R_e
mu_e = EARTH.mu_e
J2 = EARTH.J2
c = PHYS.c
CG5_a = CG5.a
CG5_b = CG5.b
CG5_c = CG5.c

__all__ = [
    "PhysicalConstants",
    "EarthConstants",
    "TimeConstants",
    "DefaultStates",
    "CG5Coefficients",
    "PHYS",
    "EARTH",
    "TIME",
    "DEFAULT",
    "CG5",
]
