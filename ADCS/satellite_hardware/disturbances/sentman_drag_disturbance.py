from __future__ import annotations
__all__ = ["Sentman_Drag_Disturbance"]

import numpy as np
from scipy.special import erf
from typing import TYPE_CHECKING, Dict

from ADCS.state import State
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.disturbances.helpers.geometry_config import GeometryConfig
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

if TYPE_CHECKING:
    from ADCS.satellite_hardware.satellite.satellite import Satellite

_BOLTZMANN = 1.380649e-23   # J/K
_AMU = 1.66053906660e-27    # kg
_SQRT_PI = np.sqrt(np.pi)


class Sentman_Drag_Disturbance(Disturbance):
    r"""
    **Free-molecular drag at finite speed ratio (Sentman's flat-plate model).**

    :class:`~ADCS.satellite_hardware.disturbances.drag_disturbance.Drag_Disturbance`
    treats the incoming air as a cold beam: every molecule arrives exactly
    along the velocity, so a face feels nothing once it is edge-on to the flow
    and the drag coefficient is a number you supply.  In the thermosphere the
    molecules also have thermal motion.  How much depends on the **speed
    ratio**

    .. math::

        S = \frac{V}{\sqrt{2 k T_\infty / m}},

    the orbital speed divided by the most probable thermal speed of the gas.
    At 7.6 km/s and exospheric temperatures of 600 K to 1500 K, :math:`S`
    runs from roughly 10 down to 6, so a fraction of the molecules reach
    faces that the cold-beam model says are shadowed, and an edge-on panel
    still feels a shear force.  That "thermal floor" is what sets the
    aerodynamic authority of a feathered solar array, and it changes with the
    solar cycle at constant density.

    This class uses Sentman's (1961) one-sided flat-plate coefficients with
    diffuse re-emission, in the form given by Moe & Moe (2005) and Doornbos
    (2012, section 3.4).  For a face with unit normal :math:`\mathbf{n}_i`,
    flow direction :math:`\hat{\mathbf{v}}` (the spacecraft velocity relative
    to the air, body frame) and :math:`\gamma_i = \mathbf{n}_i \cdot
    \hat{\mathbf{v}}`, :math:`\ell_i = \sqrt{1 - \gamma_i^2}`:

    .. math::

        P_i = \frac{e^{-\gamma_i^2 S^2}}{S}, \quad
        G = \frac{1}{2 S^2}, \quad
        Q = 1 + G, \quad
        Z_i = 1 + \operatorname{erf}(\gamma_i S),

    .. math::

        C_{D,i} = \frac{P_i}{\sqrt{\pi}} + \gamma_i Q Z_i
                  + \frac{\gamma_i}{2}\frac{v_{re}}{V}\left(\sqrt{\pi}\gamma_i Z_i + P_i\right),
        \qquad
        C_{L,i} = \ell_i G Z_i
                  + \frac{\ell_i}{2}\frac{v_{re}}{V}\left(\sqrt{\pi}\gamma_i Z_i + P_i\right),

    with the re-emission speed ratio from the energy accommodation
    coefficient :math:`\alpha` and wall temperature :math:`T_w`
    (Koppenwallner):

    .. math::

        \frac{v_{re}}{V} = \sqrt{\tfrac{1}{2}\left[1 + \alpha\left(\frac{4 k T_w}{m V^2} - 1\right)\right]}.

    The force on face :math:`i` is

    .. math::

        \mathbf{F}_i = \tfrac{1}{2}\rho V^2 A_i
        \left(-C_{D,i}\hat{\mathbf{v}} + C_{L,i}\hat{\mathbf{l}}_i\right),
        \qquad
        \hat{\mathbf{l}}_i = \frac{\gamma_i\hat{\mathbf{v}} - \mathbf{n}_i}{\ell_i},

    so the "lift" is the part of the surface pressure perpendicular to the
    flow (it points into the body, away from the face normal).  The torque is
    :math:`\sum_i (\mathbf{r}_i - \mathbf{r}_{COM}) \times \mathbf{F}_i`.

    Faces are one-sided, exactly as in ``Drag_Disturbance``: describe a closed
    body with outward normals, and a thin panel as a front/back pair.  There
    is no clipping; a back-facing face is suppressed smoothly by
    :math:`Z_i \to 0` as :math:`\gamma_i S \to -\infty`, which is what makes
    the model differentiable and gives the edge-on floor.

    In the cold-beam limit :math:`S \to \infty` with no re-emission the force
    reduces to ``Drag_Disturbance`` with :math:`C_D = 2` on every face;
    this is checked by the tests.

    **Derivatives.**  The force depends on attitude only through the
    body-frame velocity :math:`\mathbf{V}_b`.  The attitude Jacobian and
    Hessian apply the chain rule analytically through
    :math:`\partial\mathbf{V}_b/\partial\mathbf{q}` and
    :math:`\partial^2\mathbf{V}_b/\partial\mathbf{q}^2` from the orbital state;
    the velocity derivatives of the Sentman coefficients are central
    differences on :math:`\mathbf{V}_b` (three-dimensional, cheap, accurate to
    about :math:`10^{-8}` relative).

    :param config: Face geometry.  Uses ``area``, ``centroid`` and ``normal``
        of each face; the ``CD`` field is ignored (the coefficient is computed).
    :type config: :class:`~ADCS.satellite_hardware.disturbances.helpers.geometry_config.GeometryConfig`
    :param T_inf: Exospheric (gas) temperature [K]; sets the speed ratio.
    :type T_inf: float
    :param T_wall: Surface temperature [K] for the re-emitted molecules.
    :type T_wall: float
    :param alpha: Energy accommodation coefficient, 0 to 1 (0.93 is a common
        LEO value, Moe & Moe 2005).
    :type alpha: float
    :param m_amu: Mean molecular mass of the gas [amu]; 16 for atomic oxygen,
        which dominates between roughly 200 km and 600 km.
    :type m_amu: float
    """

    def __init__(
        self,
        config: GeometryConfig,
        T_inf: float = 900.0,
        T_wall: float = 300.0,
        alpha: float = 0.93,
        m_amu: float = 16.0,
    ):
        self.config = config
        params = self.config.params
        self.numfaces = len(params)
        self.areas = np.array([p["area"] for p in params], dtype=float)
        self.centroids = np.vstack([p["centroid"] for p in params]).astype(float)
        self.normals = np.vstack([normalize(p["normal"]) for p in params]).astype(float)
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError("alpha must be between 0 and 1")
        if T_inf <= 0.0 or T_wall < 0.0 or m_amu <= 0.0:
            raise ValueError("T_inf and m_amu must be positive, T_wall non-negative")
        self.T_inf = float(T_inf)
        self.T_wall = float(T_wall)
        self.alpha = float(alpha)
        self.m_amu = float(m_amu)
        super().__init__()

    # ----------------------------------------------------------------- physics
    @property
    def molecular_mass(self) -> float:
        """Mean molecular mass [kg]."""
        return self.m_amu * _AMU

    def speed_ratio(self, speed: float) -> float:
        r"""Speed ratio :math:`S = V / \sqrt{2 k T_\infty / m}` for a speed in m/s."""
        return float(speed / np.sqrt(2.0 * _BOLTZMANN * self.T_inf / self.molecular_mass))

    def reemission_ratio(self, speed: float) -> float:
        r"""Re-emission speed over incoming speed, :math:`v_{re}/V`, for a speed in m/s."""
        thermal = 4.0 * _BOLTZMANN * self.T_wall / (self.molecular_mass * speed * speed)
        return float(np.sqrt(0.5 * (1.0 + self.alpha * (thermal - 1.0))))

    def face_coefficients(self, V_B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Per-face drag and lift coefficients :math:`(C_{D,i}, C_{L,i})`.

        :param V_B: Body-frame velocity relative to the air [m/s], shape ``(3,)``.
        :return: Two arrays of shape ``(numfaces,)``, referenced to each face's own area.
        """
        V_B = np.asarray(V_B, dtype=float).reshape(3)
        speed = float(np.linalg.norm(V_B))
        if speed == 0.0:
            return np.zeros(self.numfaces), np.zeros(self.numfaces)
        vhat = V_B / speed
        S = self.speed_ratio(speed)
        vre = self.reemission_ratio(speed)
        gamma = self.normals @ vhat
        ell = np.sqrt(np.maximum(0.0, 1.0 - gamma * gamma))
        P = np.exp(-(gamma * S) ** 2) / S
        G = 1.0 / (2.0 * S * S)
        Q = 1.0 + G
        Z = 1.0 + erf(gamma * S)
        reemit = 0.5 * vre * (_SQRT_PI * gamma * Z + P)
        C_D = P / _SQRT_PI + gamma * Q * Z + gamma * reemit
        C_L = ell * G * Z + ell * reemit
        return C_D, C_L

    def face_forces(self, V_B: np.ndarray, rho: float) -> np.ndarray:
        r"""Force on every face [N], shape ``(numfaces, 3)``, body frame."""
        V_B = np.asarray(V_B, dtype=float).reshape(3)
        speed = float(np.linalg.norm(V_B))
        if rho <= 0.0 or speed == 0.0:
            return np.zeros((self.numfaces, 3))
        vhat = V_B / speed
        C_D, C_L = self.face_coefficients(V_B)
        gamma = self.normals @ vhat
        ell = np.sqrt(np.maximum(0.0, 1.0 - gamma * gamma))
        # Lift direction (gamma vhat - n)/ell; where ell == 0 the lift
        # coefficient is zero too, so any finite direction will do.
        lift_dir = gamma[:, None] * vhat[None, :] - self.normals
        safe_ell = np.where(ell > 0.0, ell, 1.0)
        lift_dir = lift_dir / safe_ell[:, None]
        q = 0.5 * rho * speed * speed * self.areas
        return q[:, None] * (-C_D[:, None] * vhat[None, :] + C_L[:, None] * lift_dir)

    # --------------------------------------------------------------- interface
    @staticmethod
    def _velocity_and_density(x: State, os: Orbital_State | Dict[str, np.ndarray]):
        vecs = os.get_state_vector(x=x)
        V_B = np.asarray(vecs["v"], dtype=float).reshape(3) * 1000.0  # km/s -> m/s
        return vecs, V_B, float(vecs["rho"])

    def force(self, sat: Satellite, x: State, os: Orbital_State | Dict[str, np.ndarray]) -> np.ndarray:
        r"""Total aerodynamic force in the body frame [N], shape ``(3,)``."""
        _, V_B, rho = self._velocity_and_density(x, os)
        return self.face_forces(V_B, rho).sum(axis=0)

    def torque(self, sat: Satellite, x: State, os: Orbital_State | Dict[str, np.ndarray]) -> np.ndarray:
        r"""Total aerodynamic torque about the centre of mass [N·m], shape ``(3,)``."""
        _, V_B, rho = self._velocity_and_density(x, os)
        return self._torque_from_velocity(V_B, rho, sat.COM)

    def _torque_from_velocity(self, V_B: np.ndarray, rho: float, COM: np.ndarray) -> np.ndarray:
        forces = self.face_forces(V_B, rho)
        return np.sum(np.cross(self.centroids - COM, forces), axis=0)

    # Velocity derivatives by central differences; the attitude chain rule is analytic.
    def _velocity_jacobian(self, fun, V_B: np.ndarray, eps: float) -> np.ndarray:
        """``out[a, k] = d fun_k / d V_a``, shape ``(3, 3)``."""
        out = np.zeros((3, 3))
        for a in range(3):
            d = np.zeros(3)
            d[a] = eps
            out[a] = (fun(V_B + d) - fun(V_B - d)) / (2.0 * eps)
        return out

    def _velocity_hessian(self, fun, V_B: np.ndarray, eps: float) -> np.ndarray:
        """``out[a, b, k] = d^2 fun_k / dV_a dV_b``, shape ``(3, 3, 3)``."""
        out = np.zeros((3, 3, 3))
        for a in range(3):
            d = np.zeros(3)
            d[a] = eps
            out[a] = (self._velocity_jacobian(fun, V_B + d, eps) - self._velocity_jacobian(fun, V_B - d, eps)) / (2.0 * eps)
        return out

    @staticmethod
    def _fd_step(V_B: np.ndarray, order: int) -> float:
        speed = max(float(np.linalg.norm(V_B)), 1.0)
        return speed * (1.0e-6 if order == 1 else 1.0e-4)

    def _attitude_jacobian(self, fun, x: State, os: Orbital_State) -> np.ndarray:
        vecs, V_B, rho = self._velocity_and_density(x, os)
        if rho <= 0.0:
            return np.zeros((4, 3))
        dV__dq = np.asarray(vecs["dv"], dtype=float).reshape(4, 3) * 1000.0
        dF__dV = self._velocity_jacobian(lambda v: fun(v, rho), V_B, self._fd_step(V_B, 1))
        return dV__dq @ dF__dV

    def _attitude_hessian(self, fun, x: State, os: Orbital_State) -> np.ndarray:
        vecs, V_B, rho = self._velocity_and_density(x, os)
        if rho <= 0.0:
            return np.zeros((4, 4, 3))
        dV__dq = np.asarray(vecs["dv"], dtype=float).reshape(4, 3) * 1000.0
        ddV__dqdq = np.asarray(vecs["ddv"], dtype=float).reshape(4, 4, 3) * 1000.0
        dF__dV = self._velocity_jacobian(lambda v: fun(v, rho), V_B, self._fd_step(V_B, 1))
        ddF__dVdV = self._velocity_hessian(lambda v: fun(v, rho), V_B, self._fd_step(V_B, 2))
        return (
            np.einsum("ija,ak->ijk", ddV__dqdq, dF__dV)
            + np.einsum("ia,jb,abk->ijk", dV__dq, dV__dq, ddF__dVdV)
        )

    def force_qjac(self, sat: Satellite, x: State, os: Orbital_State) -> np.ndarray:
        r"""Jacobian of :meth:`force` with respect to the quaternion, shape ``(4, 3)``."""
        return self._attitude_jacobian(lambda v, rho: self.face_forces(v, rho).sum(axis=0), x, os)

    def force_qqhess(self, sat: Satellite, x: State, os: Orbital_State) -> np.ndarray:
        r"""Hessian of :meth:`force` with respect to the quaternion, shape ``(4, 4, 3)``."""
        return self._attitude_hessian(lambda v, rho: self.face_forces(v, rho).sum(axis=0), x, os)

    def torque_qjac(self, sat: Satellite, x: State, os: Orbital_State) -> np.ndarray:
        r"""Jacobian of :meth:`torque` with respect to the quaternion, shape ``(4, 3)``."""
        COM = sat.COM
        return self._attitude_jacobian(lambda v, rho: self._torque_from_velocity(v, rho, COM), x, os)

    def torque_qqhess(self, sat: Satellite, x: State, os: Orbital_State) -> np.ndarray:
        r"""Hessian of :meth:`torque` with respect to the quaternion, shape ``(4, 4, 3)``."""
        COM = sat.COM
        return self._attitude_hessian(lambda v, rho: self._torque_from_velocity(v, rho, COM), x, os)
