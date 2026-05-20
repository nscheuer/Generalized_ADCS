from __future__ import annotations 
__all__ = ["SRP_Disturbance"]

import numpy as np
from numba import njit
from typing import TYPE_CHECKING
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.disturbances.helpers.geometry_config import GeometryConfig
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize, normed_vec_jac, normed_vec_hess
from ADCS.orbits.universal_constants import EarthConstants


@njit(cache=True)
def _srp_torque_kernel(S_B, R_B, COM, normals, areas, centroids,
                       eta_s, eta_d, eta_a, solar_constant, c_light):
    """SRP torque summed over geometry faces."""
    diff = np.empty(3)
    diff[0] = S_B[0] - R_B[0]
    diff[1] = S_B[1] - R_B[1]
    diff[2] = S_B[2] - R_B[2]
    d_norm = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
    inv_d = 1.0 / d_norm
    sb0 = diff[0] * inv_d
    sb1 = diff[1] * inv_d
    sb2 = diff[2] * inv_d
    nf = normals.shape[0]
    t0 = 0.0
    t1 = 0.0
    t2 = 0.0
    for i in range(nf):
        cos_g = normals[i, 0]*sb0 + normals[i, 1]*sb1 + normals[i, 2]*sb2
        if cos_g < 0.0:
            cos_g = 0.0
        A_cg = areas[i] * cos_g
        c0 = centroids[i, 0] - COM[0]
        c1 = centroids[i, 1] - COM[1]
        c2 = centroids[i, 2] - COM[2]
        ms = A_cg * (eta_a[i] + eta_d[i])
        cs0 = c1*sb2 - c2*sb1
        cs1 = c2*sb0 - c0*sb2
        cs2 = c0*sb1 - c1*sb0
        mn = A_cg * (2.0*eta_s[i]*cos_g + (2.0/3.0)*eta_d[i])
        cn0 = c1*normals[i, 2] - c2*normals[i, 1]
        cn1 = c2*normals[i, 0] - c0*normals[i, 2]
        cn2 = c0*normals[i, 1] - c1*normals[i, 0]
        t0 += ms*cs0 + mn*cn0
        t1 += ms*cs1 + mn*cn1
        t2 += ms*cs2 + mn*cn2
    k = -(solar_constant / c_light)
    out = np.empty(3)
    out[0] = k * t0
    out[1] = k * t1
    out[2] = k * t2
    return out


if TYPE_CHECKING:
    from ADCS.satellite_hardware.satellite.satellite import Satellite

class SRP_Disturbance(Disturbance):
    r"""
    **Solar Radiation Pressure (SRP) Disturbance Torque Model**

    This class models the disturbance torque caused by **solar radiation pressure (SRP)**,
    i.e., the momentum flux of sunlight impinging on and reflecting from spacecraft
    exterior surfaces.

    A discrete set of planar faces is used to represent the spacecraft geometry. For
    each face that is illuminated (Sun-facing), a force is computed using a simplified
    optical reflection model with **absorptive**, **diffuse**, and **specular** components.
    The net SRP torque is the sum of moments of these per-face forces about the spacecraft
    center of mass (COM).

    Surface geometry and optical parameters are provided by:

    :class:`~ADCS.satellite_hardware.disturbances.helpers.geometry_config.GeometryConfig`

    The Sun vector and orbital state are provided by:

    :class:`~ADCS.orbits.orbital_state.Orbital_State`

    **Reference Vectors**

    Let:

    - :math:`\mathbf{S}_B` be the Sun position/direction vector expressed in the body frame
      (as returned by :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`)
    - :math:`\mathbf{R}_B` be the spacecraft position vector (Earth to spacecraft) expressed
      in the body frame

    The implemented model forms the body-frame unit Sun-look direction:

    .. math::

        \mathbf{s}_b
        =
        \frac{\mathbf{S}_B - \mathbf{R}_B}{\left\lVert \mathbf{S}_B - \mathbf{R}_B \right\rVert},

    using :func:`~ADCS.helpers.math_helpers.normalize`.

    **Illumination Gate**

    SRP is applied only if the spacecraft is illuminated. Illumination is checked via:

    :meth:`~ADCS.orbits.orbital_state.Orbital_State.is_sunlit`

    If the spacecraft is in eclipse (not sunlit), the SRP torque is identically zero.

    **Optical Coefficients**

    Each face :math:`i=1,\dots,N` has:

    - area :math:`A_i` [m²]
    - centroid :math:`\mathbf{r}_i` (body frame) [m]
    - unit normal :math:`\mathbf{n}_i` (body frame)
    - coefficients :math:`\eta_{s,i}`, :math:`\eta_{d,i}`, :math:`\eta_{a,i}`

    The coefficients represent specular, diffuse, and absorptive contributions, respectively.
    They are typically constrained such that :math:`\eta_{s,i} + \eta_{d,i} + \eta_{a,i} = 1`,
    though this class does not enforce the constraint.

    **Radiation Pressure**

    The solar radiation pressure magnitude is modeled using the solar constant :math:`S_0`
    and the speed of light :math:`c`, both provided by:

    :class:`~ADCS.orbits.universal_constants.EarthConstants`

    .. math::

        P_\odot = \frac{S_0}{c}.

    **Implemented Force/Torque Form**

    Define the incidence cosine:

    .. math::

        \cos\gamma_i = \mathbf{n}_i^\top \mathbf{s}_b,
        \qquad
        \cos^+\gamma_i = \max(0,\cos\gamma_i).

    The implementation uses two scalar multipliers:

    .. math::

        m_{s,i} = A_i(\eta_{a,i}+\eta_{d,i})\,\cos^+\gamma_i,

    .. math::

        m_{n,i}
        =
        A_i\left(2\,\eta_{s,i}\,(\cos^+\gamma_i)^2 + \tfrac{2}{3}\,\eta_{d,i}\right)\cos^+\gamma_i,

    matching the code’s use of ``proj_area = A_i cos^+γ_i`` and
    ``m_n = proj_area * (2 η_s cos^+γ_i + 2/3 η_d)``.

    Let the lever arm from COM to the face centroid be:

    .. math::

        \mathbf{c}_i = \mathbf{r}_i - \mathbf{r}_{\mathrm{COM}}.

    The total SRP torque (body frame) is:

    .. math::

        \mathbf{T}_{\mathrm{SRP}}
        =
        -\frac{S_0}{c}\sum_{i=1}^N
        \Big[
            m_{s,i}\,(\mathbf{c}_i \times \mathbf{s}_b)
            +
            m_{n,i}\,(\mathbf{c}_i \times \mathbf{n}_i)
        \Big].

    **Differentiability**

    The clamping :math:`\cos^+\gamma_i = \max(0,\cos\gamma_i)` introduces a kink at
    :math:`\cos\gamma_i = 0`. The Jacobian and Hessian methods provided by this class
    use piecewise derivatives with a Heaviside gate (subgradient-like behavior).

    :param config: Surface geometry and optical coefficients for each face.
    :type config: :class:`~ADCS.satellite_hardware.disturbances.helpers.geometry_config.GeometryConfig`
    """

    def __init__(self, config: GeometryConfig):
        r"""
        Initialize the SRP disturbance model.

        This constructor reads per-face geometry and optical parameters from the
        provided configuration. Normals are normalized using
        :func:`~ADCS.helpers.math_helpers.normalize`.

        The following arrays are constructed:

        +----------------+----------------------------------------------+-------------+
        | Attribute      | Meaning                                      | Shape       |
        +================+==============================================+=============+
        | ``areas``      | Face areas :math:`A_i`                       | ``(N,)``    |
        +----------------+----------------------------------------------+-------------+
        | ``centroids``  | Face centroids :math:`\mathbf{r}_i`          | ``(N,3)``   |
        +----------------+----------------------------------------------+-------------+
        | ``normals``    | Unit face normals :math:`\mathbf{n}_i`       | ``(N,3)``   |
        +----------------+----------------------------------------------+-------------+
        | ``eta_s``      | Specular coefficients :math:`\eta_{s,i}`     | ``(N,)``    |
        +----------------+----------------------------------------------+-------------+
        | ``eta_d``      | Diffuse coefficients :math:`\eta_{d,i}`      | ``(N,)``    |
        +----------------+----------------------------------------------+-------------+
        | ``eta_a``      | Absorptive coefficients :math:`\eta_{a,i}`   | ``(N,)``    |
        +----------------+----------------------------------------------+-------------+

        :param config: Configuration instance describing face geometry and optical coefficients.
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
        self.eta_s = np.array([p["eta_s"] for p in params])
        self.eta_d = np.array([p["eta_d"] for p in params])
        self.eta_a = np.array([p["eta_a"] for p in params])

        super().__init__()

    def torque(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **SRP torque** in the body frame.

        This method obtains the Sun and spacecraft position vectors in the body frame via:

        :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`

        using keys:

        - ``"s"`` : :math:`\mathbf{S}_B`
        - ``"r"`` : :math:`\mathbf{R}_B`

        The unit Sun-look direction is:

        .. math::

            \mathbf{s}_b
            =
            \frac{\mathbf{S}_B - \mathbf{R}_B}{\left\lVert \mathbf{S}_B - \mathbf{R}_B \right\rVert}.

        Incidence for each face is computed as:

        .. math::

            \cos\gamma_i = \mathbf{n}_i^\top \mathbf{s}_b,
            \qquad
            \cos^+\gamma_i = \max(0,\cos\gamma_i).

        Using the per-face multipliers:

        .. math::

            m_{s,i} = A_i(\eta_{a,i}+\eta_{d,i})\,\cos^+\gamma_i,

        .. math::

            m_{n,i}
            =
            A_i\left(2\,\eta_{s,i}\,(\cos^+\gamma_i)^2 + \tfrac{2}{3}\,\eta_{d,i}\right)\cos^+\gamma_i,

        and lever arms :math:`\mathbf{c}_i = \mathbf{r}_i - \mathbf{r}_{\mathrm{COM}}`,
        the torque is:

        .. math::

            \mathbf{T}_{\mathrm{SRP}}
            =
            -\frac{S_0}{c}\sum_{i=1}^N
            \Big[
                m_{s,i}\,(\mathbf{c}_i \times \mathbf{s}_b)
                +
                m_{n,i}\,(\mathbf{c}_i \times \mathbf{n}_i)
            \Big].

        If the spacecraft is not illuminated (``os.is_sunlit()`` is false), then:

        .. math::

            \mathbf{T}_{\mathrm{SRP}} = \mathbf{0}.

        :param sat: Satellite instance providing the COM position via ``sat.COM``.
        :type sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :param x: Full spacecraft state containing the attitude quaternion used by the orbital-state mapping.
        :type x: :class:`numpy.ndarray`
        :param os: Orbital state provider supplying ``"s"``, ``"r"``, and illumination status.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: SRP disturbance torque in the body frame [N·m], shape ``(3,)``.
        :rtype: :class:`numpy.ndarray`
        """
        if not os.is_sunlit():
            return np.zeros(3)

        vecs = os.get_state_vector(x=x)
        return _srp_torque_kernel(
            vecs["s"], vecs["r"], sat.COM,
            self.normals, self.areas, self.centroids,
            self.eta_s, self.eta_d, self.eta_a,
            EarthConstants.solar_constant, EarthConstants.c,
        )

    def torque_qjac(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **Jacobian of SRP torque** with respect to the attitude quaternion.

        Let :math:`\mathbf{q}` denote the attitude quaternion inside :math:`x`. The SRP torque
        depends on :math:`\mathbf{q}` through the body-frame Sun-look direction:

        .. math::

            \mathbf{s}_b(\mathbf{q})
            =
            \frac{\mathbf{S}_B(\mathbf{q}) - \mathbf{R}_B(\mathbf{q})}
                 {\left\lVert \mathbf{S}_B(\mathbf{q}) - \mathbf{R}_B(\mathbf{q}) \right\rVert}.

        The Jacobian of :math:`\mathbf{s}_b` is computed using:

        :func:`~ADCS.helpers.math_helpers.normed_vec_jac`

        on the vector :math:`\mathbf{S}_B - \mathbf{R}_B`, with quaternion derivatives
        provided by the orbital state.

        **Clamped incidence derivative**

        For each face:

        .. math::

            \cos\gamma_i(\mathbf{q}) = \mathbf{n}_i^\top \mathbf{s}_b(\mathbf{q}),
            \qquad
            \cos^+\gamma_i(\mathbf{q}) = \max(0,\cos\gamma_i(\mathbf{q})).

        Using the Heaviside function :math:`H(\cdot)`:

        .. math::

            \frac{\partial \cos^+\gamma_i}{\partial \mathbf{q}}
            =
            H(\cos\gamma_i)\,
            \mathbf{n}_i^\top
            \frac{\partial \mathbf{s}_b}{\partial \mathbf{q}}.

        **Derivative of per-face multipliers**

        With:

        .. math::

            m_{s,i} = A_i(\eta_{a,i}+\eta_{d,i})\,\cos^+\gamma_i,

        .. math::

            m_{n,i}
            =
            A_i\left(2\,\eta_{s,i}\,(\cos^+\gamma_i)^2 + \tfrac{2}{3}\,\eta_{d,i}\right)\cos^+\gamma_i,

        the derivatives are:

        .. math::

            \frac{\partial m_{s,i}}{\partial \mathbf{q}}
            =
            A_i(\eta_{a,i}+\eta_{d,i})
            \frac{\partial \cos^+\gamma_i}{\partial \mathbf{q}},

        .. math::

            \frac{\partial m_{n,i}}{\partial \mathbf{q}}
            =
            A_i\left(
                6\eta_{s,i}(\cos^+\gamma_i)^2
                + \tfrac{2}{3}\eta_{d,i}
            \right)
            \frac{\partial \cos^+\gamma_i}{\partial \mathbf{q}},

        where the factor arises from differentiating the code-equivalent scalar form
        :math:`m_{n,i} = A_i\cos^+\gamma_i\left(2\eta_{s,i}\cos^+\gamma_i + \tfrac{2}{3}\eta_{d,i}\right)`.

        **Torque Jacobian**

        With :math:`\mathbf{c}_i = \mathbf{r}_i - \mathbf{r}_{\mathrm{COM}}`:

        .. math::

            \frac{\partial \mathbf{T}_{\mathrm{SRP}}}{\partial \mathbf{q}}
            =
            -\frac{S_0}{c}\sum_{i=1}^N
            \Big[
                \frac{\partial m_{s,i}}{\partial \mathbf{q}}(\mathbf{c}_i\times\mathbf{s}_b)
                + m_{s,i}\,\mathbf{c}_i\times\frac{\partial \mathbf{s}_b}{\partial \mathbf{q}}
                + \frac{\partial m_{n,i}}{\partial \mathbf{q}}(\mathbf{c}_i\times\mathbf{n}_i)
            \Big].

        If the spacecraft is not illuminated, the Jacobian is returned as zero.

        :param sat: Satellite instance providing the COM position via ``sat.COM``.
        :type sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :param x: Full spacecraft state containing the attitude quaternion.
        :type x: :class:`numpy.ndarray`
        :param os: Orbital state provider supplying vectors and quaternion derivatives via
                   :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`
                   (typically ``"s"``, ``"r"``, ``"ds"``, ``"dr"``).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Quaternion Jacobian ``∂T_SRP/∂q``, shape ``(3, 4)``.
        :rtype: :class:`numpy.ndarray`
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

        # d(n_i . s_b)/dq for every face: (4, Nfaces) = [quat, face]. The
        # Heaviside incidence mask is PER FACE (axis 1), not per quaternion
        # component (axis 0); the old loop iterated range(Nfaces) but indexed
        # dcos_gamma__dq[i] along the size-4 quaternion axis -- masking the
        # wrong axis (and out of range for >4 faces).
        normals_term = ds_body__dq @ self.normals.T
        face_lit = (cos_gamma > 0.0)
        dcos_gamma__dq = normals_term * face_lit[np.newaxis, :]

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
            # Canonical Jacobian layout is (4,3) = [quat, torque-component];
            # the eclipse return must match (was (3,1)).
            return np.zeros((4, 3))

    def torque_qqhess(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **Hessian of SRP torque** with respect to the attitude quaternion.

        This method returns the second derivative tensor:

        .. math::

            \frac{\partial^2 \mathbf{T}_{\mathrm{SRP}}}{\partial \mathbf{q}^2}
            \in \mathbb{R}^{3\times 4\times 4}.

        The second-order dependence arises through:

        - :math:`\mathbf{s}_b(\mathbf{q})` (a normalized vector)
        - the clamped incidence :math:`\cos^+\gamma_i(\mathbf{q})`

        The second derivative of the normalized Sun-look direction is computed using:

        :func:`~ADCS.helpers.math_helpers.normed_vec_hess`.

        **Second-order structure**

        Let :math:`\mathbf{c}_i = \mathbf{r}_i - \mathbf{r}_{\mathrm{COM}}` and define:

        .. math::

            \mathbf{t}_{s,i} = \mathbf{c}_i\times\mathbf{s}_b,
            \qquad
            \mathbf{t}_{n,i} = \mathbf{c}_i\times\mathbf{n}_i.

        Then:

        .. math::

            \mathbf{T}_{\mathrm{SRP}}
            =
            -\frac{S_0}{c}\sum_{i=1}^N
            \bigl(m_{s,i}\mathbf{t}_{s,i} + m_{n,i}\mathbf{t}_{n,i}\bigr).

        Differentiating twice and collecting terms yields the schematic form:

        .. math::

            \frac{\partial^2 \mathbf{T}_{\mathrm{SRP}}}{\partial \mathbf{q}^2}
            =
            -\frac{S_0}{c}\sum_{i=1}^N\Big[
              \frac{\partial^2 m_{s,i}}{\partial \mathbf{q}^2}\,\mathbf{t}_{s,i}
              + \frac{\partial m_{s,i}}{\partial \mathbf{q}}\otimes
                \frac{\partial \mathbf{t}_{s,i}}{\partial \mathbf{q}}
              + \left(\cdot\right)^\top
              + m_{s,i}\,\frac{\partial^2 \mathbf{t}_{s,i}}{\partial \mathbf{q}^2}
              + \frac{\partial^2 m_{n,i}}{\partial \mathbf{q}^2}\,\mathbf{t}_{n,i}
            \Big],

        where :math:`(\cdot)^\top` denotes the symmetric transpose term with swapped
        quaternion indices.

        **Clamping and gating**

        The clamped cosine introduces a kink at :math:`\cos\gamma_i = 0`. The code
        implements a piecewise (Heaviside-gated) second derivative consistent with:

        .. math::

            \frac{\partial \cos^+\gamma_i}{\partial \mathbf{q}}
            =
            H(\cos\gamma_i)\,
            \mathbf{n}_i^\top \frac{\partial \mathbf{s}_b}{\partial \mathbf{q}},
            \qquad
            \frac{\partial^2 \cos^+\gamma_i}{\partial \mathbf{q}^2}
            \approx
            H(\cos\gamma_i)\,
            \mathbf{n}_i^\top \frac{\partial^2 \mathbf{s}_b}{\partial \mathbf{q}^2},

        i.e., distributional terms at the kink are neglected.

        If the spacecraft is not illuminated, the Hessian is returned as zero.

        :param sat: Satellite instance providing the COM position via ``sat.COM``.
        :type sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :param x: Full spacecraft state containing the attitude quaternion.
        :type x: :class:`numpy.ndarray`
        :param os: Orbital state provider supplying vectors and quaternion derivatives up to second order via
                   :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`
                   (typically ``"s"``, ``"r"``, ``"ds"``, ``"dr"``, ``"dds"``, ``"ddr"``).
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Quaternion Hessian tensor ``∂²T_SRP/∂q²``, shape ``(3, 4, 4)``.
        :rtype: :class:`numpy.ndarray`
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
