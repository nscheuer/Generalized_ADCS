from __future__ import annotations 
__all__ = ["Drag_Disturbance"]

import numpy as np
import time
from typing import TYPE_CHECKING
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.disturbances.helpers.geometry_config import GeometryConfig
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

if TYPE_CHECKING:
    from ADCS.satellite_hardware.satellite.satellite import Satellite

class Drag_Disturbance(Disturbance):
    r"""
    **Aerodynamic Drag Disturbance Model**

    This class models the **aerodynamic drag torque** acting on a satellite in
    low Earth orbit (LEO). A discrete set of planar surface elements (faces) is
    used to approximate the spacecraft geometry. For each face, a drag force is
    computed from the relative atmospheric flow expressed in the **body frame**,
    and the net disturbance torque is obtained by summing the moments of the
    face forces about the spacecraft center of mass (COM).

    **Geometry and Parameters**

    The disturbance is parameterized by a configuration object:

    :class:`~ADCS.satellite_hardware.disturbances.helpers.geometry_config.GeometryConfig`

    which provides, for each face :math:`i = 1,\dots,N`:

    - area :math:`A_i` [m²]
    - centroid position :math:`\mathbf{r}_i` in the body frame [m]
    - outward unit normal :math:`\mathbf{n}_i` in the body frame (unitless)
    - drag coefficient :math:`C_{D,i}` (unitless)

    The satellite COM is provided by:

    :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
    via ``sat.COM``.

    **Relative Flow**

    The body-frame relative velocity vector :math:`\mathbf{V}_b` and density
    :math:`\rho` are obtained from the orbital state interface:

    :class:`~ADCS.orbits.orbital_state.Orbital_State`

    specifically via ``os.get_state_vector(x=...)``.

    **Face Drag Force**

    For a face with normal :math:`\mathbf{n}_i`, define the (clipped) incidence term:

    .. math::

        s_i \;=\; \max\!\bigl(0,\ \mathbf{n}_i^\top \mathbf{V}_b \bigr).

    The implemented per-face drag force is collinear with the relative flow and
    acts opposite the flow direction. Using the standard dynamic-pressure scaling,
    a compact model consistent with the implementation is:

    .. math::

        \mathbf{F}_i
        \;=\;
        -\tfrac{1}{2}\,\rho\,C_{D,i}\,A_i\,
        s_i\,\frac{\mathbf{V}_b}{\|\mathbf{V}_b\|}.

    (Some implementations absorb the normalization by :math:`\|\mathbf{V}_b\|`
    into other terms. Here, :math:`s_i` is linear in :math:`\mathbf{V}_b` and the
    net torque expression below is written to match the code path.)

    **Net Drag Torque**

    Let :math:`\mathbf{r}_{\mathrm{COM}}` be the spacecraft COM in body coordinates.
    The total drag torque about COM is:

    .. math::

        \mathbf{T}_{\mathrm{drag}}
        \;=\;
        \sum_{i=1}^N
        \bigl(\mathbf{r}_i - \mathbf{r}_{\mathrm{COM}}\bigr)\times \mathbf{F}_i.

    In the form used by the implementation, define

    .. math::

        F_i \;=\; C_{D,i}A_i\,\max\!\bigl(0,\mathbf{n}_i^\top\mathbf{V}_b\bigr),

    stack lever arms :math:`\mathbf{c}_i = \mathbf{r}_i - \mathbf{r}_{\mathrm{COM}}`,
    and write:

    .. math::

        \mathbf{T}_{\mathrm{drag}}
        \;=\;
        -\tfrac{1}{2}\rho
        \left(\sum_{i=1}^N F_i\,\mathbf{c}_i\right)\times\mathbf{V}_b.

    **Differentiability**

    The incidence clipping :math:`\max(0,\cdot)` introduces a kink at
    :math:`\mathbf{n}_i^\top\mathbf{V}_b = 0`. The Jacobian and Hessian provided by
    :meth:`~ADCS.satellite_hardware.disturbances.drag_disturbance.Drag_Disturbance.torque_qjac`
    and
    :meth:`~ADCS.satellite_hardware.disturbances.drag_disturbance.Drag_Disturbance.torque_qqhess`
    follow the piecewise derivatives using a Heaviside-gated term. At the kink,
    derivatives are understood in the piecewise (one-sided) sense used by the code.

    :param config: Geometry and aerodynamic coefficients for each face.
    :type config: :class:`~ADCS.satellite_hardware.disturbances.helpers.geometry_config.GeometryConfig`
    """
    def __init__(self, config: GeometryConfig):
        r"""
        Initialize the aerodynamic drag disturbance model from a geometry configuration.

        This constructor reads the per-face parameters from
        :class:`~ADCS.satellite_hardware.disturbances.helpers.geometry_config.GeometryConfig`,
        normalizes the provided face normals using
        :func:`~ADCS.helpers.math_helpers.normalize`, and stores:

        - ``numfaces``: number of modeled faces :math:`N`
        - ``areas``: :math:`A_i` for each face, shape ``(N,)``
        - ``centroids``: :math:`\mathbf{r}_i` for each face, shape ``(N,3)``
        - ``normals``: unit normals :math:`\mathbf{n}_i`, shape ``(N,3)``
        - ``CDs``: drag coefficients :math:`C_{D,i}`, shape ``(N,)``

        :param config: Configuration containing face areas, centroids, normals, and drag coefficients.
        :type config: :class:`~ADCS.satellite_hardware.disturbances.helpers.geometry_config.GeometryConfig`
        :return: None
        :rtype: None
        """
        self.config = config
        params = self.config.params

        self.numfaces = len(params)
        self.areas = np.array([p["area"] for p in params])
        self.centroids = np.vstack([p["centroid"] for p in params])
        self.normals = np.vstack([normalize(p["normal"]) for p in params])
        self.CDs = np.array([p["CD"] for p in params])
        
        super().__init__()

    def torque(self, sat: Satellite, x: np.ndarray, os: Orbital_State | Dict[str, np.ndarray]) -> np.ndarray:
        r"""
        Compute the **aerodynamic drag torque** in the body frame.

        The orbital state interface is queried as:

        .. math::

            \{\mathbf{V}_b,\rho\} \leftarrow \texttt{os.get\_state\_vector}(x),

        where :math:`\mathbf{V}_b` is the relative velocity in the **body frame**
        and :math:`\rho` is the atmospheric density.

        If the orbital-state velocity is provided in km/s, it is converted to m/s
        prior to use:

        .. math::

            \mathbf{V}_b[\mathrm{m/s}] \;=\; 1000\,\mathbf{V}_b[\mathrm{km/s}].

        For each face, define the clipped incidence scalar:

        .. math::

            s_i \;=\; \max\!\bigl(0,\ \mathbf{n}_i^\top \mathbf{V}_b\bigr),

        and the scalar face factor:

        .. math::

            F_i \;=\; C_{D,i}A_i\,s_i.

        Let the lever arm from COM to face centroid be:

        .. math::

            \mathbf{c}_i \;=\; \mathbf{r}_i - \mathbf{r}_{\mathrm{COM}}.

        The implemented net torque is:

        .. math::

            \mathbf{T}_{\mathrm{drag}}
            \;=\;
            -\tfrac{1}{2}\rho
            \left(\sum_{i=1}^N F_i\,\mathbf{c}_i\right)\times\mathbf{V}_b.

        **Units**

        - :math:`\rho` in kg/m³
        - :math:`\mathbf{V}_b` in m/s
        - :math:`A_i` in m²
        - torque output in N·m

        :param sat: Satellite instance providing the center of mass via ``sat.COM``.
        :type sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :param x: Full spacecraft state; must contain the attitude quaternion used by the orbital-state mapping.
        :type x: :class:`numpy.ndarray`
        :param os: Orbital state provider that supplies ``"v"`` (body-frame relative velocity) and ``"rho"`` (density)
                   through ``os.get_state_vector(x=...)``. Some applications may pass a mapping with equivalent keys.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State` | :class:`dict`\\[:class:`str`, :class:`numpy.ndarray`\\]
        :return: Total aerodynamic drag torque in the body frame, shape ``(3,)``.
        :rtype: :class:`numpy.ndarray`
        """
        vecs = os.get_state_vector(x=x)

        V_B = vecs["v"]*1000.0 #m/s
        rho = vecs["rho"]

        v_proj = np.maximum(0, np.dot(self.normals, V_B))
        F = self.CDs*self.areas*v_proj
        cents = self.centroids - sat.COM
        ct = 0.5*rho
        torque = -ct*np.cross(F@cents, V_B)

        t1 = time.process_time()
        
        return torque
    
    def torque_qjac(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **Jacobian of drag torque with respect to the attitude quaternion**.

        The drag torque depends on the attitude quaternion :math:`\mathbf{q}` (contained in :math:`x`)
        through the body-frame flow :math:`\mathbf{V}_b(\mathbf{q})`. The orbital state provider is
        expected to return:

        - :math:`\mathbf{V}_b` via ``"v"``
        - :math:`\rho` via ``"rho"``
        - :math:`\dfrac{\partial \mathbf{V}_b}{\partial \mathbf{q}}` via ``"dv"``

        from:

        :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`.

        **Piecewise derivative of incidence**

        For each face, define:

        .. math::

            s_i(\mathbf{q}) = \max\!\bigl(0,\ \mathbf{n}_i^\top \mathbf{V}_b(\mathbf{q})\bigr).

        Using the Heaviside step function :math:`H(\cdot)`:

        .. math::

            \frac{\partial s_i}{\partial \mathbf{q}}
            \;=\;
            H\!\bigl(\mathbf{n}_i^\top \mathbf{V}_b\bigr)\,
            \mathbf{n}_i^\top \frac{\partial \mathbf{V}_b}{\partial \mathbf{q}}.

        Then:

        .. math::

            F_i(\mathbf{q}) = C_{D,i}A_i\,s_i(\mathbf{q}),
            \qquad
            \frac{\partial F_i}{\partial \mathbf{q}}
            = C_{D,i}A_i\,\frac{\partial s_i}{\partial \mathbf{q}}.

        **Jacobian of torque**

        With:

        .. math::

            \mathbf{C}(\mathbf{q}) = \sum_{i=1}^N F_i(\mathbf{q})\,\mathbf{c}_i,

        the torque is:

        .. math::

            \mathbf{T}_{\mathrm{drag}}(\mathbf{q})
            = -\tfrac{1}{2}\rho\ \mathbf{C}(\mathbf{q}) \times \mathbf{V}_b(\mathbf{q}).

        Applying the product rule:

        .. math::

            \frac{\partial \mathbf{T}_{\mathrm{drag}}}{\partial \mathbf{q}}
            =
            -\tfrac{1}{2}\rho\left[
              \left(\frac{\partial \mathbf{C}}{\partial \mathbf{q}}\right)\times\mathbf{V}_b
              +
              \mathbf{C}\times\left(\frac{\partial \mathbf{V}_b}{\partial \mathbf{q}}\right)
            \right].

        The returned Jacobian has shape ``(3,4)`` corresponding to derivatives of the 3 torque
        components with respect to the 4 quaternion components.

        :param sat: Satellite instance providing the center of mass via ``sat.COM``.
        :type sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :param x: Full spacecraft state containing the attitude quaternion.
        :type x: :class:`numpy.ndarray`
        :param os: Orbital state provider supplying ``"v"``, ``"rho"``, and ``"dv"`` through
                   :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Quaternion Jacobian :math:`\partial \mathbf{T}_{\mathrm{drag}}/\partial \mathbf{q}`,
                 shape ``(3, 4)``.
        :rtype: :class:`numpy.ndarray`
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

        This method returns the second derivative tensor of the drag torque with respect to the
        quaternion components, with shape ``(3, 4, 4)``. The orbital state provider is expected
        to return, via
        :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`:

        - :math:`\mathbf{V}_b` via ``"v"``
        - :math:`\rho` via ``"rho"``
        - :math:`\dfrac{\partial \mathbf{V}_b}{\partial \mathbf{q}}` via ``"dv"``
        - :math:`\dfrac{\partial^2 \mathbf{V}_b}{\partial \mathbf{q}^2}` via ``"ddv"``

        **Face factors and their derivatives**

        For each face:

        .. math::

            s_i(\mathbf{q}) = \max\!\bigl(0,\ \mathbf{n}_i^\top \mathbf{V}_b(\mathbf{q})\bigr),
            \qquad
            F_i(\mathbf{q}) = C_{D,i}A_i\,s_i(\mathbf{q}).

        Using the Heaviside gate for the first derivative:

        .. math::

            \frac{\partial s_i}{\partial \mathbf{q}}
            =
            H\!\bigl(\mathbf{n}_i^\top \mathbf{V}_b\bigr)\,
            \mathbf{n}_i^\top \frac{\partial \mathbf{V}_b}{\partial \mathbf{q}}.

        Likewise, the second derivative is taken piecewise (gated) as implemented:

        .. math::

            \frac{\partial^2 s_i}{\partial \mathbf{q}^2}
            \approx
            H\!\bigl(\mathbf{n}_i^\top \mathbf{V}_b\bigr)\,
            \mathbf{n}_i^\top \frac{\partial^2 \mathbf{V}_b}{\partial \mathbf{q}^2},

        noting that distributional terms at the kink are neglected by this piecewise model.

        Therefore:

        .. math::

            \frac{\partial F_i}{\partial \mathbf{q}} = C_{D,i}A_i\,\frac{\partial s_i}{\partial \mathbf{q}},
            \qquad
            \frac{\partial^2 F_i}{\partial \mathbf{q}^2} = C_{D,i}A_i\,\frac{\partial^2 s_i}{\partial \mathbf{q}^2}.

        **Hessian of torque**

        Define:

        .. math::

            \mathbf{C}(\mathbf{q}) = \sum_{i=1}^N F_i(\mathbf{q})\,\mathbf{c}_i.

        The torque is:

        .. math::

            \mathbf{T}_{\mathrm{drag}}(\mathbf{q})
            = -\tfrac{1}{2}\rho\ \mathbf{C}(\mathbf{q})\times \mathbf{V}_b(\mathbf{q}).

        The second derivative (arranged as a tensor over quaternion components) follows from
        differentiating the Jacobian and grouping the mixed terms:

        .. math::

            \frac{\partial^2 \mathbf{T}_{\mathrm{drag}}}{\partial \mathbf{q}^2}
            =
            -\tfrac{1}{2}\rho\Big[
              \left(\frac{\partial^2 \mathbf{C}}{\partial \mathbf{q}^2}\right)\times\mathbf{V}_b
              +
              \left(\frac{\partial \mathbf{C}}{\partial \mathbf{q}}\right)\times
              \left(\frac{\partial \mathbf{V}_b}{\partial \mathbf{q}}\right)
              +
              \left(\left(\frac{\partial \mathbf{C}}{\partial \mathbf{q}}\right)\times
              \left(\frac{\partial \mathbf{V}_b}{\partial \mathbf{q}}\right)\right)^{\!\top}
              +
              \mathbf{C}\times\left(\frac{\partial^2 \mathbf{V}_b}{\partial \mathbf{q}^2}\right)
            \Big],

        where :math:`(\cdot)^{\top}` denotes the symmetric pairing term obtained by swapping the
        quaternion-derivative indices (as implemented via an explicit transpose of the mixed term).

        :param sat: Satellite instance providing the center of mass via ``sat.COM``.
        :type sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :param x: Full spacecraft state containing the attitude quaternion.
        :type x: :class:`numpy.ndarray`
        :param os: Orbital state provider supplying ``"v"``, ``"rho"``, ``"dv"``, and ``"ddv"`` through
                   :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Quaternion Hessian tensor :math:`\partial^2 \mathbf{T}_{\mathrm{drag}}/\partial \mathbf{q}^2`,
                 shape ``(3, 4, 4)``.
        :rtype: :class:`numpy.ndarray`
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


