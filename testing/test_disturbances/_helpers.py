from __future__ import annotations

import numpy as np

from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.disturbances import GeometryConfig, GeometryFace
from ADCS.satellite_hardware.satellite import Satellite


def make_satellite(
    *,
    J_0: np.ndarray | None = None,
    COM: np.ndarray | None = None,
    disturbances: list | None = None,
) -> Satellite:
    return Satellite(
        J_0=np.diag([2.0, 3.0, 10.0]) if J_0 is None else np.asarray(J_0, dtype=float),
        COM=np.zeros(3) if COM is None else np.asarray(COM, dtype=float),
        disturbances=[] if disturbances is None else disturbances,
    )


def make_state(
    *,
    w: np.ndarray | None = None,
    q: np.ndarray | None = None,
) -> np.ndarray:
    omega = np.zeros(3) if w is None else np.asarray(w, dtype=float)
    quat = np.array([1.0, 0.0, 0.0, 0.0]) if q is None else normalize(np.asarray(q, dtype=float))
    return np.concatenate([omega, quat])


def make_orbital_state(
    *,
    R: np.ndarray | None = None,
    V: np.ndarray | None = None,
    B: np.ndarray | None = None,
    S: np.ndarray | None = None,
    rho: float = 5.0e-12,
    sunlit: bool | None = None,
) -> Orbital_State:
    os = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 20.0, -10.0]) if R is None else np.asarray(R, dtype=float),
        V=np.array([0.2, 8.1, -0.1]) if V is None else np.asarray(V, dtype=float),
        B=np.array([2.0e-5, -3.0e-5, 4.0e-5]) if B is None else np.asarray(B, dtype=float),
        S=np.array([1.0e8, 2.0e8, -1.0e8]) if S is None else np.asarray(S, dtype=float),
        rho=rho,
    )
    if sunlit is not None:
        os.is_sunlit = (lambda sunlit=sunlit: sunlit)
    return os


def make_geometry_config(*faces: GeometryFace) -> GeometryConfig:
    return GeometryConfig(list(faces))


def fd_quat_jac(fun, x: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    jac = np.zeros((4, 3))
    for index in range(4):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[3 + index] += eps
        x_minus[3 + index] -= eps
        x_plus[3:7] = normalize(x_plus[3:7])
        x_minus[3:7] = normalize(x_minus[3:7])
        jac[index, :] = (fun(x_plus) - fun(x_minus)) / (2.0 * eps)
    return jac


def fd_quat_hess(fun, x: np.ndarray, eps: float = 5.0e-6) -> np.ndarray:
    hess = np.zeros((4, 4, 3))
    for index in range(4):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[3 + index] += eps
        x_minus[3 + index] -= eps
        x_plus[3:7] = normalize(x_plus[3:7])
        x_minus[3:7] = normalize(x_minus[3:7])
        hess[index, :, :] = (fd_quat_jac(fun, x_plus, eps) - fd_quat_jac(fun, x_minus, eps)) / (2.0 * eps)
    return hess


def fd_vec_jac(fun, value: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    jac = np.zeros((3, 3))
    for index in range(3):
        value_plus = np.asarray(value, dtype=float).copy()
        value_minus = np.asarray(value, dtype=float).copy()
        value_plus[index] += eps
        value_minus[index] -= eps
        jac[index, :] = (fun(value_plus) - fun(value_minus)) / (2.0 * eps)
    return jac


def resolve_method(obj, *names: str):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{obj!r} does not define any of {names!r}")
