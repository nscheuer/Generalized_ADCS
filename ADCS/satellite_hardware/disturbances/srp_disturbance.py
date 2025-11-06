from __future__ import annotations 
__all__ = ["SRP_Disturbance"]

import numpy as np
from typing import TYPE_CHECKING
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.disturbances.geometry_config import GeometryConfig
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize, normed_vec_jac, normed_vec_hess
from ADCS.orbits.universal_constants import EarthConstants


if TYPE_CHECKING:
    from ADCS.satellite_hardware.satellite.satellite import Satellite

class SRP_Disturbance(Disturbance):
    r"""
    **Solar Radiation Pressure (SRP) Disturbance Model**

    This class models the **solar radiation pressure (SRP)** torque acting on a
    satellite. The SRP torque arises from the momentum exchange between sunlight
    and the satellite’s external surfaces due to reflection and absorption.

    The model computes the total torque by summing contributions from all
    illuminated surfaces defined in a :class:`~ADCS.satellite_hardware.disturbances.geometry_config.GeometryConfig`.

    **Physical Model**

    The solar radiation pressure force on a surface element is given by:

    .. math::

        \mathbf{F}_i =
        -P_{\odot} A_i \max(0, \mathbf{n}_i \cdot \mathbf{s}_b)
        \left[
            (1 - \eta_s) \mathbf{s}_b
            + 2 \eta_s (\mathbf{n}_i \cdot \mathbf{s}_b) \mathbf{n}_i
            + \tfrac{2}{3} \eta_d \mathbf{n}_i
        \right]

    where:

    - :math:`P_{\odot} = \dfrac{S_0}{c}` is the solar radiation pressure [N/m²],
    - :math:`S_0` — Solar constant (:math:`1367~\mathrm{W/m^2}`),
    - :math:`c` — Speed of light in vacuum [m/s],
    - :math:`A_i` — Area of the i-th surface [m²],
    - :math:`\mathbf{n}_i` — Surface normal vector in body frame,
    - :math:`\mathbf{s}_b` — Unit Sun direction vector in body frame,
    - :math:`\eta_s, \eta_d, \eta_a` — Specular, diffuse, and absorptive coefficients.

    The total torque on the spacecraft is:

    .. math::

        \mathbf{T}_{\mathrm{SRP}} =
        \sum_i (\mathbf{r}_i - \mathbf{r}_{\mathrm{COM}}) \times \mathbf{F}_i

    Only surfaces facing the Sun (:math:`\mathbf{n}_i \cdot \mathbf{s}_b > 0`) contribute
    to the disturbance. If the spacecraft is in eclipse, the torque is zero.

    Parameters
    ----------
    config : :class:`~ADCS.satellite_hardware.disturbances.geometry_config.GeometryConfig`
        Configuration containing surface properties (areas, centroids, normals,
        and optical coefficients).

    Attributes
    ----------
    numfaces : int
        Number of modeled surface elements.

    areas : :class:`numpy.ndarray`
        Surface areas of each face [m²], shape ``(N,)``.

    centroids : :class:`numpy.ndarray`
        Surface centroids in body coordinates [m], shape ``(N, 3)``.

    normals : :class:`numpy.ndarray`
        Surface normal unit vectors in body frame, shape ``(N, 3)``.

    eta_s, eta_d, eta_a : :class:`numpy.ndarray`
        Specular, diffuse, and absorptive reflection coefficients.
    """

    def __init__(self, config: GeometryConfig):
        r"""
        Initialize the SRP disturbance model.

        Parameters
        ----------
        config : :class:`~ADCS.satellite_hardware.disturbances.geometry_config.GeometryConfig`
            Configuration instance describing each satellite surface element.
        """
        self.config = config
        params = self.config.params

        self.numfaces = len(params)
        self.areas = np.array([p["area"] for p in params])
        self.centroids = np.vstack([p["centroid"] for p in params])
        self.normals = np.vstack([normalize(p["normal"]) for p in params])
        self.eta_s = np.array([p["eta_s"] for p in params])
        self.eta_d = np.array([p["eta_d"] for p in params])
        self.eta_a = np.array([p["eta_a"] for p in params])

    def torque(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **solar radiation pressure (SRP) torque** on the spacecraft in the body frame.

        The implemented model forms the Sun-look direction
        :math:`\mathbf{s}_b = \frac{\mathbf{S}_b - \mathbf{R}_b}{\lVert \mathbf{S}_b - \mathbf{R}_b\rVert}`
        (using :func:`~ADCS.helpers.math_helpers.normalize`) and uses
        :math:`\cos\gamma_i = \mathbf{n}_i^\top \mathbf{s}_b`. With the *clamped* incidence
        :math:`\cos^+ \gamma_i = \max(0,\cos\gamma_i)`, the per-face scalar multipliers are

        .. math::

            m_{s,i} \;=\; A_i\,(\eta_a+\eta_d)\,\cos^+\gamma_i, \qquad
            m_{n,i} \;=\; A_i\!\left(2\,\eta_s\,(\cos^+\gamma_i)^2 + \tfrac{2}{3}\,\eta_d\right),

        and the SRP torque is

        .. math::

            \mathbf{T}_{\mathrm{SRP}}
            \;=\;
            -\frac{S_0}{c}\;
            \sum_i \Big[
                m_{s,i}\,(\mathbf{r}_i-\mathbf{r}_{\mathrm{COM}})\times \mathbf{s}_b
                \;+\;
                m_{n,i}\,(\mathbf{r}_i-\mathbf{r}_{\mathrm{COM}})\times \mathbf{n}_i
            \Big],

        where :math:`S_0` is the solar constant and :math:`c` the speed of light.

        Illumination is checked via :meth:`~ADCS.orbits.orbital_state.Orbital_State.is_sunlit`;
        if the spacecraft is not sunlit, the torque is zero.

        Parameters
        ----------
        sat : :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
            Satellite object providing the center-of-mass ``sat.COM`` (body frame) and geometry.
        x : :class:`numpy.ndarray`
            Full spacecraft state (contains the attitude quaternion used to form body-frame vectors).
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Provides :math:`\mathbf{S}_b` (Sun vector), :math:`\mathbf{R}_b` (spacecraft position),
            and illumination status.

        Returns
        -------
        :class:`numpy.ndarray`
            SRP torque vector in the body frame [N·m], shape ``(3,)``.
        """
        vecs = os.get_state_vector(x=x)

        S_B = vecs["s"]
        R_B = vecs["r"]

        s_body = normalize(S_B - R_B)

        cos_gamma = np.maximum(0, np.dot(self.normals, s_body))
        proj_area = self.areas * cos_gamma
        cents = self.centroids - sat.COM

        m_s = proj_area * (self.eta_a + self.eta_d)
        t_s = m_s @ np.cross(cents, s_body)

        m_n = proj_area * (2 * self.eta_s * cos_gamma + (2 / 3) * self.eta_d)
        t_n = m_n @ np.cross(cents, self.normals)

        if os.is_sunlit():
            return -(EarthConstants.solar_constant / EarthConstants.c) * (t_s + t_n)
        else:
            return np.zeros((3, 1))

    def torque_qjav(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian of the **SRP torque** with respect to the **attitude quaternion**.

        Let :math:`\mathbf{q}` be the attitude quaternion inside :math:`x`.
        Define :math:`\mathbf{s}_b(\mathbf{q}) = \frac{\mathbf{S}_b-\mathbf{R}_b}{\lVert \mathbf{S}_b-\mathbf{R}_b\rVert}`
        and :math:`\cos\gamma_i(\mathbf{q})=\mathbf{n}_i^\top\mathbf{s}_b(\mathbf{q})`.
        Using the clamped incidence :math:`\cos^+\gamma_i=\max(0,\cos\gamma_i)`, its derivative is

        .. math::

            \frac{\partial \cos^+\gamma_i}{\partial \mathbf{q}}
            \;=\;
            H\!\big(\cos\gamma_i\big)\;
            \mathbf{n}_i^\top\frac{\partial \mathbf{s}_b}{\partial \mathbf{q}},

        where :math:`H(\cdot)` is the Heaviside function (subgradient at zero).
        With

        .. math::

            m_{s,i} = A_i(\eta_a+\eta_d)\,\cos^+\gamma_i, \qquad
            m_{n,i} = A_i\!\left(2\,\eta_s(\cos^+\gamma_i)^2 + \tfrac{2}{3}\eta_d\right),

        the Jacobian follows from the product rule

        .. math::

            \frac{\partial \mathbf{T}_{\mathrm{SRP}}}{\partial \mathbf{q}}
            \;=\;
            -\frac{S_0}{c}\sum_i \Big[
                \frac{\partial m_{s,i}}{\partial \mathbf{q}}\,
                (\mathbf{r}_i-\mathbf{r}_{\mathrm{COM}})\times \mathbf{s}_b
                \;+\;
                m_{s,i}\,(\mathbf{r}_i-\mathbf{r}_{\mathrm{COM}})\times
                \frac{\partial \mathbf{s}_b}{\partial \mathbf{q}}
                \;+\;
                \frac{\partial m_{n,i}}{\partial \mathbf{q}}\,
                (\mathbf{r}_i-\mathbf{r}_{\mathrm{COM}})\times \mathbf{n}_i
            \Big],

        with

        .. math::

            \frac{\partial m_{s,i}}{\partial \mathbf{q}}
            = A_i(\eta_a+\eta_d)\,H(\cos\gamma_i)\,
            \mathbf{n}_i^\top\frac{\partial \mathbf{s}_b}{\partial \mathbf{q}}, \qquad
            \frac{\partial m_{n,i}}{\partial \mathbf{q}}
            = A_i\Big[
                4\eta_s\,\cos^+\gamma_i\,H(\cos\gamma_i)\,
                \mathbf{n}_i^\top\frac{\partial \mathbf{s}_b}{\partial \mathbf{q}}
            \Big].

        In code, :math:`\frac{\partial \mathbf{s}_b}{\partial \mathbf{q}}`
        is produced via :func:`~ADCS.helpers.math_helpers.normed_vec_jac`
        applied to :math:`\mathbf{S}_b-\mathbf{R}_b`.

        Parameters
        ----------
        sat : :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
            Satellite instance providing the COM location.
        x : :class:`numpy.ndarray`
            Full spacecraft state containing the quaternion :math:`\mathbf{q}`.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Provides :math:`\mathbf{S}_b, \mathbf{R}_b` and their quaternion derivatives.

        Returns
        -------
        :class:`numpy.ndarray`
            Jacobian matrix ``∂T_SRP/∂q``, shape ``(3, 4)``.
        """
        # (Implementation same as in your code; mathematical explanation retained)
        vecs = os.get_state_vector(x=x)
        S_B = vecs["s"]
        R_B = vecs["r"]

        s_body = normalize(S_B - R_B)
        ds_body__dq = normed_vec_jac(S_B - R_B, vecs["ds"] - vecs["dr"])

        cos_gamma = np.dot(self.normals, s_body)
        for i in range(len(cos_gamma)):
            if cos_gamma[i] < 0:
                cos_gamma[i] = 0.0

        proj_area = self.areas * cos_gamma
        cents = self.centroids - sat.COM

        dcos_gamma__dq = np.zeros_like(ds_body__dq @ self.normals.T)
        for i in range(len(cos_gamma)):
            if cos_gamma[i] > 0:
                dcos_gamma__dq[i] = (ds_body__dq @ self.normals.T)[i]
            else:
                dcos_gamma__dq[i] = 0.0

        dproj_area__dq = self.areas * dcos_gamma__dq

        m_s = proj_area * (self.eta_a + self.eta_d)
        dm_s__dq = dproj_area__dq * (self.eta_a + self.eta_d)
        dt_s__dq = dm_s__dq @ np.cross(cents, s_body) + np.cross(m_s @ cents, ds_body__dq)

        dm_n__dq = (
            dproj_area__dq * (2 * self.eta_s * cos_gamma + (2 / 3) * self.eta_d)
            + proj_area * (2 * self.eta_s * dcos_gamma__dq)
        )
        dt_n__dq = dm_n__dq @ np.cross(cents, self.normals)

        if os.is_sunlit():
            return -(EarthConstants.solar_constant / EarthConstants.c) * (dt_s__dq + dt_n__dq)
        else:
            return np.zeros((3, 1))

    def torque_qqhess(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Hessian of the **SRP torque** with respect to the **attitude quaternion**.

        Let :math:`\mathbf{q}` be the attitude quaternion and abbreviate
        :math:`\mathbf{r}_i'=\mathbf{r}_i-\mathbf{r}_{\mathrm{COM}}`,
        :math:`\mathbf{s}_b=\mathbf{s}_b(\mathbf{q})`,
        :math:`\cos^+\gamma_i=\max(0,\mathbf{n}_i^\top\mathbf{s}_b)`.
        Using the Jacobian in :meth:`torque_qjav` and the product rule, the second derivative is

        .. math::

            \frac{\partial^2 \mathbf{T}_{\mathrm{SRP}}}{\partial \mathbf{q}^2}
            \;=\;
            -\frac{S_0}{c}\sum_i \Big[
                \frac{\partial^2 m_{s,i}}{\partial \mathbf{q}^2}\,
                (\mathbf{r}_i'\times \mathbf{s}_b)
                \;+\;
                2\,\frac{\partial m_{s,i}}{\partial \mathbf{q}}\times
                \frac{\partial (\mathbf{r}_i'\times \mathbf{s}_b)}{\partial \mathbf{q}}
                \;+\;
                m_{s,i}\,\mathbf{r}_i'\times
                \frac{\partial^2 \mathbf{s}_b}{\partial \mathbf{q}^2}
                \;+\;
                \frac{\partial^2 m_{n,i}}{\partial \mathbf{q}^2}\,
                (\mathbf{r}_i'\times \mathbf{n}_i)
            \Big],

        where the terms involving :math:`\cos^+\gamma_i` inherit a Heaviside factor
        :math:`H(\cos\gamma_i)` and are undefined exactly at :math:`\cos\gamma_i=0`
        (a measure-zero set in practice). The implementation follows this structure by
        building first- and second-order derivatives of the clamped incidence and of
        :math:`\mathbf{s}_b(\mathbf{q})`.

        Parameters
        ----------
        sat : :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
            Satellite instance providing the COM location.
        x : :class:`numpy.ndarray`
            Full spacecraft state containing the quaternion :math:`\mathbf{q}`.
        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Provides :math:`\mathbf{S}_b, \mathbf{R}_b` and their first/second quaternion derivatives.

        Returns
        -------
        :class:`numpy.ndarray`
            Quaternion Hessian tensor of the SRP torque, shape ``(3, 4, 4)``.
        """
        # (Implementation as in your original code)
        vecs = os.get_state_vector(x=x)

        S_B = vecs["s"]
        R_B = vecs["r"]

        s_body = normalize(S_B - R_B)
        ds_body__dq = normed_vec_jac(S_B - R_B, vecs["ds"] - vecs["dr"])
        dds_body__dqdq = normed_vec_hess(S_B - R_B, vecs["ds"] - vecs["dr"], vecs["dds"] - vecs["ddr"])

        cos_gamma = np.maximum(0, np.dot(self.normals, s_body))
        proj_area = self.areas * cos_gamma
        cents = self.centroids - sat.COM

        dcos_gamma__dq = (cos_gamma > 0) * (ds_body__dq @ self.normals.T)
        ddcos_gamma__dqdq = (cos_gamma > 0) * (dds_body__dqdq @ self.normals.T)

        dproj_area__dq = self.areas * dcos_gamma__dq
        ddproj_area__dqdq = self.areas * ddcos_gamma__dqdq

        m_s = proj_area * (self.eta_a + self.eta_d)
        dm_s__dq = dproj_area__dq * (self.eta_a + self.eta_d)
        ddm_s__dqdq = ddproj_area__dqdq * (self.eta_a + self.eta_d)

        dt_s__dq = dm_s__dq @ np.cross(cents, s_body) + np.cross(m_s @ cents, ds_body__dq)
        tmp = np.cross(np.expand_dims(dm_s__dq @ cents, 0), np.expand_dims(ds_body__dq, 1))
        ddt_s__dqdq = (
            ddm_s__dqdq @ np.cross(cents, s_body)
            + tmp
            + np.transpose(tmp, (1, 0, 2))
            + np.cross(m_s @ cents, dds_body__dqdq)
        )

        dm_n__dq = dproj_area__dq * (2 * self.eta_s * cos_gamma + (2 / 3) * self.eta_d) + proj_area * (2 * self.eta_s * dcos_gamma__dq)
        tmp2 = np.expand_dims(dproj_area__dq, 0) * np.expand_dims((2 * self.eta_s * dcos_gamma__dq), 1)
        ddm_n__dqdq = (
            ddproj_area__dqdq * (2 * self.eta_s * cos_gamma + (2 / 3) * self.eta_d)
            + tmp2
            + np.transpose(tmp2, (1, 0, 2))
            + proj_area * (2 * self.eta_s * ddcos_gamma__dqdq)
        )
        ddt_n__dqdq = ddm_n__dqdq @ np.cross(cents, self.normals)

        if os.is_sunlit():
            return -(EarthConstants.solar_constant / EarthConstants.c) * (ddt_s__dqdq + ddt_n__dqdq)
        else:
            return np.zeros((3, 4, 4))
