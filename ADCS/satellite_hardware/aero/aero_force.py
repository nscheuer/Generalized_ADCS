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
    :param Cn: Normal (pressure) coefficient (default 2.0).
    :param Ct: Tangential (shear) coefficient (default 0.0). ``Cn == Ct`` gives a
        lift-free (drag-only) model.
    """

    def __init__(self, normals, areas, Cn: float = 2.0, Ct: float = 0.0) -> None:
        self.normals = np.vstack([normalize(np.asarray(n, dtype=float)) for n in normals])
        self.areas = np.asarray(areas, dtype=float).reshape(-1)
        self.Cn = float(Cn)
        self.Ct = float(Ct)

    @classmethod
    def from_geometry(cls, config: GeometryConfig, Cn: float = 2.0, Ct: float = 0.0) -> "AeroModel":
        r"""
        Build an :class:`AeroModel` from a satellite :class:`GeometryConfig`
        (the same per-face geometry used by the drag-torque disturbance).
        """
        params = config.params
        normals = [p["normal"] for p in params]
        areas = [p["area"] for p in params]
        return cls(normals=normals, areas=areas, Cn=Cn, Ct=Ct)

    def force_body(self, V_b, rho) -> np.ndarray:
        r"""
        Net body-frame aerodynamic force [N].

        :param V_b: Body-frame relative wind velocity [m/s], shape ``(3,)``.
        :param rho: Atmospheric density [kg/m^3].
        """
        return panel_aero_force_body(V_b, rho, self.normals, self.areas, self.Cn, self.Ct)
