from __future__ import annotations
__all__ = ["AeroModel", "panel_aero_force_body"]

# finite speed-ratio (Storch 2002) companion model:
# ADCS.satellite_hardware.aero.finite_s

import numpy as np

from ADCS.helpers.math_helpers import normalize
from ADCS.satellite_hardware.disturbances.helpers.geometry_config import GeometryConfig


def panel_aero_force_body(V_b, rho, normals, areas, Cn, Ct):
    r"""
    Free-molecular panel aerodynamic force in the body frame (drag + lift).

    For each surface panel with outward unit normal :math:`\hat{n}` and area
    :math:`A`, exposed to a body-frame relative wind velocity
    :math:`\mathbf{V}_b` (magnitude :math:`V`, direction :math:`\hat{v}`), define
    the (clipped) incidence cosine :math:`c = \max(0,\hat{n}\cdot\hat{v})`. The
    panel force combines a **normal pressure** term (momentum delivered along the
    surface normal) and a **tangential shear** term (momentum delivered along the
    in-plane flow direction):

    .. math::

        \mathbf{F} = -\tfrac{1}{2}\,\rho V^2 A\, c\,
            \Big[\, C_n\, c\, \hat{n} \;+\; C_t\,(\hat{v} - c\,\hat{n}) \,\Big].

    The **standard** dynamic pressure :math:`\tfrac{1}{2}\rho V^2` is used,
    matching the drag-torque kernel's :math:`-\tfrac{1}{2}\rho` and the
    free-molecular literature, so :math:`C_n` is the normal-incidence drag
    coefficient (~2.0-2.4 for a diffuse plate). At normal incidence
    (:math:`c=1`, :math:`\hat n=\hat v`) the force is purely along
    :math:`-\hat v` (drag, zero lift). At oblique incidence the normal-
    pressure term contributes a component perpendicular to :math:`\hat v` — the
    **lift** — whose magnitude scales with :math:`(C_n-C_t)`; setting
    :math:`C_n=C_t` recovers a lift-free (drag-only) panel. Faces in the wake
    (:math:`c\le 0`) contribute nothing.

    :param V_b: Body-frame relative wind velocity [m/s], shape ``(3,)``.
    :param rho: Atmospheric density [kg/m^3].
    :param normals: Unit outward normals, shape ``(M, 3)``.
    :param areas: Panel areas [m^2], shape ``(M,)``.
    :param Cn: Normal (pressure) coefficient.
    :param Ct: Tangential (shear) coefficient.
    :return: Net aerodynamic force in the body frame [N], shape ``(3,)``.
    """
    V_b = np.asarray(V_b, dtype=float).reshape(3)
    V2 = float(V_b @ V_b)
    if rho <= 0.0 or V2 <= 0.0:
        return np.zeros(3)

    V = np.sqrt(V2)
    vhat = V_b / V
    c = normals @ vhat                      # (M,) incidence cosine per face
    c = np.where(c > 0.0, c, 0.0)           # only windward faces contribute

    scale = 0.5 * rho * V2 * np.asarray(areas, dtype=float) * c   # (M,) standard 1/2 rho V^2
    F_normal = -(Cn * scale * c)[:, None] * normals
    F_shear = -(Ct * scale)[:, None] * (vhat[None, :] - c[:, None] * normals)
    return (F_normal + F_shear).sum(axis=0)


class AeroModel:
    r"""
    Attitude-dependent orbital aerodynamic force (drag + lift) for a satellite.

    Wraps a free-molecular panel model (see :func:`panel_aero_force_body`) built
    from the satellite surface geometry. Because the force depends on the
    satellite's attitude (how each panel is oriented into the flow), it couples
    the orbit to attitude and must be evaluated in the simulation loop — see
    :class:`~ADCS.formation.constellation.Constellation` (operator-split
    co-integration). This model produces the orbital **force** only; attitude
    aerodynamic torque remains the responsibility of
    :class:`~ADCS.satellite_hardware.disturbances.drag_disturbance.Drag_Disturbance`.

    :param normals: Unit outward panel normals, shape ``(M, 3)``.
    :param areas: Panel areas [m^2], shape ``(M,)``.
    :param Cn: Normal (pressure) coefficient (default 2.0), hyperthermal mode.
    :param Ct: Tangential (shear) coefficient (default 0.0). ``Cn == Ct`` gives a
        lift-free (drag-only) model.
    :param mode: ``"hyperthermal"`` (default, unchanged behaviour) or
        ``"finite_s"`` -- the finite speed-ratio model
        (:func:`~ADCS.satellite_hardware.aero.finite_s.panel_aero_force_body_finite_s`),
        which adds the thermal floor a hyperthermal model cannot represent
        (a feathered plate still feels along-wind shear) and exposes
        exospheric temperature as an axis. Near-feather flight -- plate-built
        formation satellites, differential-drag control -- needs it; see the
        module for the accommodation parameterization.
    :param sigma_n: Normal accommodation (finite_s mode; default 0.90).
    :param sigma_t: Tangential accommodation (finite_s mode; default 0.80).
        ``(0.90, 0.80)`` maps to the hyperthermal ``(Cn, Ct) = (2.2, 1.6)``.
    :param T_wall: Surface temperature [K] (finite_s mode).
    :param T_inf: DEFAULT exospheric temperature [K] (finite_s mode); may be
        overridden per call, e.g. driven by solar activity.
    :param m_amu: Mean atmospheric molecular mass [amu] (finite_s mode).

    In ``finite_s`` mode ``normals``/``areas`` are PLATE-ONCE: list each plate
    once by either face normal (the model integrates both sides), rather than
    the +/- pairs the hyperthermal model expects.
    """

    def __init__(self, normals, areas, Cn: float = 2.0, Ct: float = 0.0,
                 mode: str = "hyperthermal", sigma_n: float = 0.90,
                 sigma_t: float = 0.80, T_wall: float = 300.0,
                 T_inf: float = 900.0, m_amu: float = 16.0) -> None:
        self.normals = np.vstack([normalize(np.asarray(n, dtype=float)) for n in normals])
        self.areas = np.asarray(areas, dtype=float).reshape(-1)
        self.Cn = float(Cn)
        self.Ct = float(Ct)
        if mode not in ("hyperthermal", "finite_s"):
            raise ValueError(f"unknown aero mode {mode!r} (hyperthermal | finite_s)")
        self.mode = mode
        self.sigma_n = float(sigma_n)
        self.sigma_t = float(sigma_t)
        self.T_wall = float(T_wall)
        self.T_inf = float(T_inf)
        self.m_amu = float(m_amu)

    @classmethod
    def from_geometry(cls, config: GeometryConfig, Cn: float = 2.0, Ct: float = 0.0,
                      **kwargs) -> "AeroModel":
        r"""
        Build an :class:`AeroModel` from a satellite :class:`GeometryConfig`
        (the same per-face geometry used by the drag-torque disturbance).
        Extra keyword arguments are forwarded to the constructor (``mode``,
        accommodation coefficients, temperatures).
        """
        params = config.params
        normals = [p["normal"] for p in params]
        areas = [p["area"] for p in params]
        return cls(normals=normals, areas=areas, Cn=Cn, Ct=Ct, **kwargs)

    def force_body(self, V_b, rho, T_inf: float = None) -> np.ndarray:
        r"""
        Net body-frame aerodynamic force [N].

        :param V_b: Body-frame relative wind velocity [m/s], shape ``(3,)``.
        :param rho: Atmospheric density [kg/m^3].
        :param T_inf: Exospheric temperature [K] for this call (``finite_s``
            mode only); defaults to the model's ``T_inf``. Solar activity
            swings it 600-1500 K, changing the near-feather floor ~58 % with
            no density change, so a solar-variable density model should drive
            this too.
        """
        if self.mode == "finite_s":
            from ADCS.satellite_hardware.aero.finite_s import (
                panel_aero_force_body_finite_s)
            return panel_aero_force_body_finite_s(
                V_b, rho, self.normals, self.areas, sigma_n=self.sigma_n,
                sigma_t=self.sigma_t, T_wall=self.T_wall,
                T_inf=self.T_inf if T_inf is None else float(T_inf),
                m_amu=self.m_amu)
        return panel_aero_force_body(V_b, rho, self.normals, self.areas, self.Cn, self.Ct)
