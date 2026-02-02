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
    """
    R_e: float = 6378.1363            # km
    R_moon: float = 1737.4            # km
    mu_e: float = 398600.4415         # km^3/s^2
    J2coeff: float = 1.082635854e-3
    J2: float = J2coeff * R_e**2 * mu_e  # km^5/s^2
    m_earth: float = 5.9736e24        # kg
    solar_constant: float = 1361.0    # W/m^2
    c: float = 299792458.0            # m/s


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
    "TimeConstants",
    "DefaultStates",
    "CG5",
]
