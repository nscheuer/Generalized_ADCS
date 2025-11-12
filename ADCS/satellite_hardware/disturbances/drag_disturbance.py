from __future__ import annotations 
__all__ = ["Drag_Disturbance"]

import numpy as np
from typing import TYPE_CHECKING
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.disturbances.geometry_config import GeometryConfig
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

if TYPE_CHECKING:
    from ADCS.satellite_hardware.satellite.satellite import Satellite

class Drag_Disturbance(Disturbance):
    r"""
    **Aerodynamic Drag Disturbance Model**

    This class models the **aerodynamic drag torque** acting on a satellite in
    low Earth orbit (LEO). The drag force on each exposed surface is computed
    from the relative atmospheric flow in the body frame, and the net disturbance
    torque is obtained as the cross product of surface forces with their lever arms
    from the satellite’s center of mass.

    **Physical Model**

    The drag force acting on a satellite surface is modeled as:

    .. math::

        \mathbf{F}_i = -\tfrac{1}{2} \rho \, C_{D,i} \, A_i \,
        \max(0, \mathbf{n}_i \cdot \mathbf{V}_b) \, \frac{\mathbf{V}_b}{\|\mathbf{V}_b\|}

    where:

    - :math:`\rho` — Atmospheric density [kg/m³]
    - :math:`C_{D,i}` — Drag coefficient of the *i*-th surface
    - :math:`A_i` — Area of the *i*-th surface [m²]
    - :math:`\mathbf{n}_i` — Surface normal (unit vector) in the body frame
    - :math:`\mathbf{V}_b` — Relative velocity vector in the body frame [m/s]

    The **total drag torque** acting on the spacecraft is:

    .. math::

        \mathbf{T}_{\mathrm{drag}}
        = \sum_i (\mathbf{r}_i - \mathbf{r}_{\mathrm{COM}}) \times \mathbf{F}_i

    where :math:`\mathbf{r}_i` is the centroid of the i-th surface and
    :math:`\mathbf{r}_{\mathrm{COM}}` is the satellite center of mass.

    Parameters
    ----------
    config : :class:`~ADCS.satellite_hardware.geometry.GeometryConfig`
        Configuration object containing face areas, centroids, normals, and drag coefficients
        for each surface element.

    Attributes
    ----------
    numfaces : int
        Number of discrete surface faces modeled.

    areas : :class:`numpy.ndarray`
        Surface areas of each face [m²], shape ``(N,)``.

    centroids : :class:`numpy.ndarray`
        Centroid positions of each face in body coordinates [m], shape ``(N, 3)``.

    normals : :class:`numpy.ndarray`
        Unit normal vectors of each face in the body frame, shape ``(N, 3)``.

    CDs : :class:`numpy.ndarray`
        Drag coefficients :math:`C_D` for each face.
    """
    def __init__(self, config: GeometryConfig):
        r"""
        Initialize the aerodynamic drag disturbance model.

        Parameters
        ----------
        config : :class:`~ADCS.satellite_hardware.geometry.GeometryConfig`
            Configuration instance providing geometric and aerodynamic parameters
            for each surface element.
        """
        self.config = config
        params = self.config.params

        self.numfaces = len(params)
        self.areas = np.array([p["area"] for p in params])
        self.centroids = np.vstack([p["centroid"] for p in params])
        self.normals = np.vstack([normalize(p["normal"]) for p in params])
        self.CDs = np.array([p["CD"] for p in params])

    def torque(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **aerodynamic drag torque** in the body frame.

        Using the force model above, the implemented torque expression is

        .. math::

            \mathbf{T}_{\mathrm{drag}}
            = -\tfrac{1}{2}\,\rho
              \sum_i C_{D,i}A_i
              \max\!\big(0,\ \mathbf{n}_i^\top\mathbf{V}_b\big)
              \big(\mathbf{r}_i-\mathbf{r}_{\mathrm{COM}}\big)\times \mathbf{V}_b.

        **State dependency.** The full state :math:`x` contains the attitude quaternion
        :math:`\mathbf{q}`; the relative flow :math:`\mathbf{V}_b=\mathbf{V}_b(\mathbf{q})`
        is provided by :class:`~ADCS.orbits.orbital_state.Orbital_State`.

        Parameters
        ----------
        sat : :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
            Satellite instance providing the COM position via ``sat.COM``.
        x : :class:`numpy.ndarray`
            Full spacecraft state; must contain the attitude quaternion.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Provides body-frame relative velocity :math:`\mathbf{V}_b` and density :math:`\rho`.

        Returns
        -------
        :class:`numpy.ndarray`
            Total aerodynamic drag torque [N·m], shape ``(3,)``.
        """
        vecs = os.get_state_vector(x=x)

        V_B = vecs["v"] * 1000.0 # km/s to m/s
        rho = vecs["rho"]

        v_proj = np.maximum(0, np.dot(self.normals, V_B))
        F = self.CDs*self.areas*v_proj
        cents = self.centroids - sat.COM
        ct = 0.5*rho
        return -ct*np.cross(F@cents, V_B)
    
    def torque_qjac(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **Jacobian of drag torque with respect to the attitude quaternion**.

        Let :math:`\mathbf{q}` be the attitude quaternion inside :math:`x`. The torque depends
        on :math:`\mathbf{q}` via :math:`\mathbf{V}_b(\mathbf{q})` and via the term
        :math:`\max(0,\,\mathbf{n}_i^\top\mathbf{V}_b)`.
        Using :math:`H(\cdot)` for the Heaviside function,

        .. math::

            \frac{\partial}{\partial \mathbf{q}}
            \max\!\big(0,\ \mathbf{n}_i^\top\mathbf{V}_b\big)
            = H\!\big(\mathbf{n}_i^\top\mathbf{V}_b\big)\,
              \mathbf{n}_i^\top\frac{\partial \mathbf{V}_b}{\partial \mathbf{q}}.

        The resulting Jacobian is consistent with the implemented expression

        .. math::

            \frac{\partial \mathbf{T}_{\mathrm{drag}}}{\partial \mathbf{q}}
            = -\tfrac{1}{2}\rho\Big[
                \big(\tfrac{\partial F}{\partial \mathbf{q}}\cdot\mathbf{r}\big)\times\mathbf{V}_b
                + \big(F\cdot\mathbf{r}\big)\times
                  \frac{\partial \mathbf{V}_b}{\partial \mathbf{q}}
            \Big],

        where :math:`F_i = C_{D,i}A_i\max(0,\mathbf{n}_i^\top\mathbf{V}_b)` and
        :math:`\mathbf{r}` stacks the lever arms :math:`(\mathbf{r}_i-\mathbf{r}_{\mathrm{COM}})`.

        Parameters
        ----------
        sat : :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
            Satellite instance providing the COM.
        x : :class:`numpy.ndarray`
            Full spacecraft state containing the attitude quaternion :math:`\mathbf{q}`.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Provides :math:`\mathbf{V}_b`, density :math:`\rho`,
            and the quaternion derivative :math:`\partial \mathbf{V}_b / \partial \mathbf{q}`.

        Returns
        -------
        :class:`numpy.ndarray`
            Quaternion Jacobian ``∂T_drag/∂q``, shape ``(3, 4)``.
        """
        vecs = os.get_state_vector(x=x)

        V_B = vecs["v"]
        rho = vecs["rho"]

        dv_body__dq = vecs["dv"]
        ddv_body__dqdq = vecs["ddv"]

        cos_alpha = np.maximum(0, np.dot(self.normals, V_B))
        F = self.CDs * self.areas * cos_alpha
        cents = self.centroids - sat.COM

        # Initialize derivative arrays
        dcos_alpha__dq = np.zeros_like(dv_body__dq @ self.normals.T)

        # Explicit conditional logic instead of inline boolean mask
        normals_term = dv_body__dq @ self.normals.T
        for i in range(len(cos_alpha)):
            if cos_alpha[i] > 0:
                dcos_alpha__dq[i] = normals_term[i]
            else:
                dcos_alpha__dq[i] = np.zeros_like(normals_term[i])

        dF__dq = dcos_alpha__dq * self.CDs * self.areas

        ct = 0.5 * rho
        return -ct * (np.cross(dF__dq @ cents, V_B) + np.cross(F @ cents, dv_body__dq)) * self.active

    def torque_qqhess(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **Hessian of drag torque with respect to the attitude quaternion**.

        With :math:`\mathbf{q}` the attitude quaternion, define
        :math:`F_i(\mathbf{q}) = C_{D,i}A_i\max(0,\mathbf{n}_i^\top\mathbf{V}_b(\mathbf{q}))`.
        Then

        .. math::

            \frac{\partial^2 \mathbf{T}_{\mathrm{drag}}}{\partial \mathbf{q}^2}
            = -\tfrac{1}{2}\rho\Big[
                \frac{\partial^2 (F\cdot\mathbf{r})}{\partial \mathbf{q}^2}\times\mathbf{V}_b
                + \frac{\partial (F\cdot\mathbf{r})}{\partial \mathbf{q}}\times
                  \frac{\partial \mathbf{V}_b}{\partial \mathbf{q}}
                + \big(\cdot\big)^\top
                + (F\cdot\mathbf{r})\times
                  \frac{\partial^2 \mathbf{V}_b}{\partial \mathbf{q}^2}
            \Big],

        where :math:`(\cdot)^\top` denotes the term with the cross-product factors swapped,
        matching the implemented symmetric pairing.

        Parameters
        ----------
        sat : :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
            Satellite instance providing the COM.
        x : :class:`numpy.ndarray`
            Full spacecraft state containing the attitude quaternion :math:`\mathbf{q}`.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Provides :math:`\mathbf{V}_b`, :math:`\rho`,
            and derivatives :math:`\partial \mathbf{V}_b / \partial \mathbf{q}`,
            :math:`\partial^2 \mathbf{V}_b / \partial \mathbf{q}^2`.

        Returns
        -------
        :class:`numpy.ndarray`
            Quaternion Hessian tensor ``∂²T_drag/∂q²``, shape ``(3, 4, 4)``.
        """
        vecs = os.get_state_vector(x=x)

        V_B = vecs["v"]
        rho = vecs["rho"]
        dv_body__dq = vecs["dv"]
        ddv_body__dqdq = vecs["ddv"]

        cos_alpha = np.maximum(0, np.dot(self.normals, V_B))
        F = self.CDs * self.areas * cos_alpha
        cents = self.centroids - sat.COM

        dcos_alpha__dq = np.zeros_like(dv_body__dq @ self.normals.T)
        ddcos_alpha__dqdq = np.zeros_like(ddv_body__dqdq @ self.normals.T)

        normals_term_1 = dv_body__dq @ self.normals.T
        normals_term_2 = ddv_body__dqdq @ self.normals.T

        for i in range(len(cos_alpha)):
            if cos_alpha[i] > 0:
                dcos_alpha__dq[i] = normals_term_1[i]
                ddcos_alpha__dqdq[i] = normals_term_2[i]
            else:
                dcos_alpha__dq[i] = np.zeros_like(normals_term_1[i])
                ddcos_alpha__dqdq[i] = np.zeros_like(normals_term_2[i])

        dF__dq = dcos_alpha__dq * self.CDs * self.areas
        ddF__dqdq = ddcos_alpha__dqdq * self.CDs * self.areas
        ct = 0.5 * rho

        tmp = np.cross(np.expand_dims(dF__dq @ cents, 0), np.expand_dims(dv_body__dq, 1))

        return -ct * (
            np.cross(ddF__dqdq @ cents, V_B)
            + tmp
            + np.transpose(tmp, (1, 0, 2))
            + np.cross(F @ cents, ddv_body__dqdq)
        )


