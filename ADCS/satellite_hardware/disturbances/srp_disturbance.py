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

    def torque(self, sat: Satellite, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **solar radiation pressure torque** on the spacecraft.

        The torque is derived as:

        .. math::

            \mathbf{T}_{\mathrm{SRP}} =
            -\frac{S_0}{c} \sum_i
            \left[
                m_{s,i} (\mathbf{r}_i - \mathbf{r}_{\mathrm{COM}}) \times \mathbf{s}_b
                + m_{n,i} (\mathbf{r}_i - \mathbf{r}_{\mathrm{COM}}) \times \mathbf{n}_i
            \right]

        where:

        - :math:`m_{s,i} = A_i (\eta_a + \eta_d) \max(0, \mathbf{n}_i \cdot \mathbf{s}_b)`
        - :math:`m_{n,i} = A_i [ 2 \eta_s \max(0, \mathbf{n}_i \cdot \mathbf{s}_b)^2 + (2/3)\eta_d ]`

        Parameters
        ----------
        sat : :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
            Satellite object containing center-of-mass position ``sat.COM``.

        q : :class:`numpy.ndarray`
            Satellite attitude quaternion (4,).

        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state providing Sun position, satellite position, and illumination status.

        Returns
        -------
        :class:`numpy.ndarray`
            SRP torque vector in body frame [N·m], shape ``(3,)``.
        """
        vecs = os.get_state_vector(q0=q)

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

    def torque_qjav(self, sat: Satellite, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **Jacobian of SRP torque** with respect to the attitude quaternion.

        The derivative accounts for the sensitivity of the Sun direction vector
        and the incidence angle to small attitude changes.

        Parameters
        ----------
        sat : :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
            Satellite instance providing the center-of-mass location.

        q : :class:`numpy.ndarray`
            Satellite attitude quaternion (4,).

        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state providing Sun and satellite positions, and their derivatives.

        Returns
        -------
        :class:`numpy.ndarray`
            Jacobian matrix of SRP torque w.r.t. quaternion, shape ``(3, 4)``.
        """
        # (Implementation same as in your code; mathematical explanation retained)
        vecs = os.get_state_vector(q0=q)
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

    def torque_qqhess(self, sat: Satellite, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **Hessian of SRP torque** with respect to the attitude quaternion.

        This derivative captures second-order sensitivity of the SRP torque to
        attitude, accounting for the curvature of the Sun vector in quaternion space.

        Parameters
        ----------
        sat : :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
            Satellite instance providing the center-of-mass location.

        q : :class:`numpy.ndarray`
            Satellite attitude quaternion (4,).

        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state providing Sun and satellite positions and their higher-order derivatives.

        Returns
        -------
        :class:`numpy.ndarray`
            Quaternion Hessian tensor of SRP torque, shape ``(3, 4, 4)``.
        """
        # (Implementation as in your original code)
        ...
