from __future__ import annotations 
__all__ = ["GG_Disturbance"]

import numpy as np
from typing import TYPE_CHECKING
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.disturbances.helpers.geometry_config import GeometryConfig
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize, normed_vec_jac, normed_vec_hess, norm, vec_norm_jac, vec_norm_hess
from ADCS.orbits.universal_constants import EarthConstants

if TYPE_CHECKING:
    from ADCS.satellite_hardware.satellite.satellite import Satellite

class GG_Disturbance(Disturbance):
    r"""
    **Gravity-Gradient Disturbance Torque Model**

    This class implements the **gravity-gradient (GG) torque** acting on a rigid
    spacecraft in orbit about the Earth. Gravity-gradient torque arises from the
    non-uniform gravitational field acting on a spacecraft with an extended mass
    distribution and is one of the dominant passive disturbance torques for
    elongated or inertia-asymmetric spacecraft in low Earth orbit (LEO).

    The model assumes:
    
    - A rigid spacecraft with constant inertia tensor expressed in the body frame
    - A central gravitational field characterized by Earth's gravitational parameter
      :math:`\mu_e`
    - The spacecraft attitude is represented implicitly through the mapping of
      orbital position into the body frame

    All quantities are computed in the **body frame**.

    **Orbital Coupling**

    The disturbance depends on the body-frame position vector of the spacecraft
    center of mass relative to the Earth, provided by:

    :class:`~ADCS.orbits.orbital_state.Orbital_State`

    via :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`.

    **Inertia Properties**

    The spacecraft inertia tensor about the center of mass is provided by:

    :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`

    via the attribute ``sat.J_0``.

    This class provides:
    
    - the gravity-gradient torque
    - its Jacobian with respect to the attitude quaternion
    - its Hessian with respect to the attitude quaternion

    These derivatives are required for high-order attitude dynamics,
    sensitivity analysis, and second-order optimization or estimation methods.
    """
    def __init__(self):
        r"""
        Initialize the gravity-gradient disturbance model.

        This constructor performs no parameter initialization beyond invoking
        the base :class:`~ADCS.satellite_hardware.disturbances.disturbance.Disturbance`
        initializer. All required physical parameters (Earth gravity constant,
        inertia tensor, orbital state) are supplied dynamically during evaluation.

        :return: None
        :rtype: None
        """
        super().__init__()

    def torque(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **gravity-gradient torque** in the body frame.

        Let :math:`\mathbf{R}_B` be the position vector from the Earth center to the
        spacecraft center of mass, expressed in the body frame. Define the unit
        radial vector:

        .. math::

            \hat{\mathbf{r}} = \frac{\mathbf{R}_B}{\|\mathbf{R}_B\|}

        and the **nadir-pointing direction**:

        .. math::

            \mathbf{n} = -\hat{\mathbf{r}}.

        The classical gravity-gradient torque acting on a rigid spacecraft is:

        .. math::

            \mathbf{T}_{\mathrm{GG}}
            =
            \frac{3\mu_e}{\|\mathbf{R}_B\|^3}
            \, \mathbf{n} \times \left( \mathbf{J}_0 \mathbf{n} \right),

        where:

        - :math:`\mu_e` is Earth’s gravitational parameter
          (:class:`~ADCS.orbits.universal_constants.EarthConstants.mu_e`)
        - :math:`\mathbf{J}_0` is the spacecraft inertia tensor about COM in the body frame

        **Dependencies**

        - :math:`\mathbf{R}_B` is obtained from
          :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`
        - The attitude quaternion contained in ``x`` determines the body-frame
          orientation of :math:`\mathbf{R}_B`

        :param sat: Satellite instance providing the inertia tensor ``sat.J_0``.
        :type sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :param x: Full spacecraft state containing the attitude quaternion.
        :type x: :class:`numpy.ndarray`
        :param os: Orbital state provider supplying the body-frame position vector ``"r"``.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Gravity-gradient torque vector in the body frame, shape ``(3,)``.
        :rtype: :class:`numpy.ndarray`
        """
        vecs = os.get_state_vector(x=x)

        R_B = vecs["r"]
        r_body_hat = normalize(R_B)
        nadir_vec = -r_body_hat

        const_term = 3.0*EarthConstants.mu_e/(norm(R_B)**3.0)
        return const_term * np.cross(nadir_vec, sat.J_0 @ nadir_vec)
    
    def torque_qvac(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **Jacobian of the gravity-gradient torque with respect to the
        attitude quaternion**.

        The gravity-gradient torque can be written as:

        .. math::

            \mathbf{T}_{\mathrm{GG}}(\mathbf{q})
            =
            c(\mathbf{q}) \,
            \mathbf{v}(\mathbf{q}),

        where:

        .. math::

            c(\mathbf{q}) = \frac{3\mu_e}{\|\mathbf{R}_B(\mathbf{q})\|^3},
            \qquad
            \mathbf{v}(\mathbf{q}) = \mathbf{n}(\mathbf{q}) \times
            \left(\mathbf{J}_0 \mathbf{n}(\mathbf{q})\right).

        The Jacobian follows from the product rule:

        .. math::

            \frac{\partial \mathbf{T}_{\mathrm{GG}}}{\partial \mathbf{q}}
            =
            \frac{\partial c}{\partial \mathbf{q}} \otimes \mathbf{v}
            +
            c \, \frac{\partial \mathbf{v}}{\partial \mathbf{q}}.

        **Intermediate derivatives**

        The unit vector derivative is computed using:

        - :func:`~ADCS.helpers.math_helpers.normed_vec_jac`
        - :func:`~ADCS.helpers.math_helpers.vec_norm_jac`

        The nadir direction derivative is:

        .. math::

            \frac{\partial \mathbf{n}}{\partial \mathbf{q}}
            =
            -\frac{\partial \hat{\mathbf{r}}}{\partial \mathbf{q}}.

        The derivative of the vector cross-product term is:

        .. math::

            \frac{\partial \mathbf{v}}{\partial \mathbf{q}}
            =
            \frac{\partial \mathbf{n}}{\partial \mathbf{q}}
            \times (\mathbf{J}_0 \mathbf{n})
            +
            \mathbf{n} \times
            \left(\mathbf{J}_0 \frac{\partial \mathbf{n}}{\partial \mathbf{q}}\right).

        The resulting Jacobian has shape ``(4,3)`` or ``(3,4)`` depending on convention;
        this implementation returns a matrix consistent with the internal ADCS
        quaternion ordering.

        :param sat: Satellite instance providing the inertia tensor ``sat.J_0``.
        :type sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :param x: Full spacecraft state containing the attitude quaternion.
        :type x: :class:`numpy.ndarray`
        :param os: Orbital state provider supplying position and its quaternion derivatives
                   via ``"r"``, ``"dr"``.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Quaternion Jacobian of gravity-gradient torque.
        :rtype: :class:`numpy.ndarray`
        """
        vecs = os.get_state_vector(x=x)

        R_B = vecs["r"]
        r_body_hat = normalize(R_B)
        dr_body_hat__dq = normed_vec_jac(R_B, vecs["dr"])
        nadir_vec = -r_body_hat
        dnadir_vec__dq = -dr_body_hat__dq

        const_term = 3.0*EarthConstants.mu_e/(norm(R_B)**3.0)

        dc__dq = -9.0*EarthConstants.mu_e*vec_norm_jac(R_B, vecs["dr"])/(norm(R_B)**4.0)
        dv__dq = (
            np.cross(dnadir_vec__dq, sat.J_0 @ nadir_vec)
            + np.cross(nadir_vec, sat.J_0 @ dnadir_vec__dq)
        )
        vec_term = np.cross(nadir_vec, sat.J_0 @ nadir_vec)

        return np.outer(dc__dq, vec_term) + const_term*dv__dq

    def torque__qqhess(self, sat: Satellite, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **Hessian of the gravity-gradient torque with respect to the
        attitude quaternion**.

        This method evaluates the second derivative tensor:

        .. math::

            \frac{\partial^2 \mathbf{T}_{\mathrm{GG}}}{\partial \mathbf{q}^2},

        which has shape ``(3,4,4)`` and is required for second-order sensitivity
        analysis, trajectory optimization, and higher-order attitude dynamics.

        Using the decomposition:

        .. math::

            \mathbf{T}_{\mathrm{GG}} = c(\mathbf{q})\,\mathbf{v}(\mathbf{q}),

        the Hessian is obtained via repeated application of the product rule:

        .. math::

            \frac{\partial^2 \mathbf{T}_{\mathrm{GG}}}{\partial \mathbf{q}^2}
            =
            \frac{\partial^2 c}{\partial \mathbf{q}^2} \otimes \mathbf{v}
            +
            \frac{\partial c}{\partial \mathbf{q}} \otimes
            \frac{\partial \mathbf{v}}{\partial \mathbf{q}}
            +
            \left(\cdot\right)^{\top}
            +
            c \, \frac{\partial^2 \mathbf{v}}{\partial \mathbf{q}^2},

        where :math:`(\cdot)^{\top}` denotes the symmetric transpose with respect to
        the quaternion derivative indices.

        **Second-order geometric derivatives**

        The implementation relies on:

        - :func:`~ADCS.helpers.math_helpers.normed_vec_hess`
        - :func:`~ADCS.helpers.math_helpers.vec_norm_hess`

        to compute second derivatives of normalized vectors and vector norms.

        The second derivative of the nadir direction is:

        .. math::

            \frac{\partial^2 \mathbf{n}}{\partial \mathbf{q}^2}
            =
            -\frac{\partial^2 \hat{\mathbf{r}}}{\partial \mathbf{q}^2}.

        Mixed cross-product terms are explicitly symmetrized to ensure consistency
        with the analytical Hessian structure.

        :param sat: Satellite instance providing the inertia tensor ``sat.J_0``.
        :type sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :param x: Full spacecraft state containing the attitude quaternion.
        :type x: :class:`numpy.ndarray`
        :param os: Orbital state provider supplying position and its first and second
                   quaternion derivatives via ``"r"``, ``"dr"``, ``"ddr"``.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`
        :return: Quaternion Hessian of gravity-gradient torque,
                 shape ``(3, 4, 4)``.
        :rtype: :class:`numpy.ndarray`
        """
        vecs = os.get_state_vector(x=x)

        R_B = vecs["r"]
        r_body_hat = normalize(R_B)
        dr_body_hat__dq = normed_vec_jac(R_B, vecs["dr"])
        ddr_body_hat__dq = normed_vec_hess(R_B, vecs["dr"], vecs["ddr"])
        nadir_vec = -r_body_hat
        dnadir_vec__dq = -dr_body_hat__dq
        ddnadir_vec__dqdq = -ddr_body_hat__dq

        const_term = 3.0*EarthConstants.mu_e/(norm(R_B)**3.0)
        tmp = np.cross(
            np.expand_dims(dnadir_vec__dq, 1),
            np.expand_dims(sat.J_0 @ dnadir_vec__dq, 0),
        )

        dc__dq = -9.0*EarthConstants.mu_e*vec_norm_jac(R_B, vecs["dr"])/(norm(R_B)**4.0)
        ddc__dqdq = (
            -9.0*EarthConstants.mu_e*vec_norm_hess(R_B, vecs["dr"], vecs["ddr"])/(norm(R_B)**4.0)
            - 4.0*np.outer(
                vec_norm_jac(R_B, vecs["dr"]),
                vec_norm_jac(R_B, vecs["dr"]),
            )/(norm(R_B)**5.0)
        )
        dv__dq = (
            np.cross(dnadir_vec__dq, sat.J_0 @ nadir_vec)
            + np.cross(nadir_vec, sat.J_0 @ dnadir_vec__dq)
        )
        ddv__dqdq = (
            np.cross(ddnadir_vec__dqdq, sat.J_0 @ nadir_vec)
            + tmp
            + np.transpose(tmp, (1, 0, 2))
            + np.cross(nadir_vec, sat.J_0 @ ddnadir_vec__dqdq)
        )

        vec_term = np.cross(nadir_vec, sat.J_0 @ nadir_vec)
        tmp2 = np.multiply.outer(dc__dq, dv__dq)

        return (
            np.multiply.outer(ddc__dqdq, vec_term)
            + tmp2
            + np.transpose(tmp2, (1, 0, 2))
            + const_term*ddv__dqdq
        )
