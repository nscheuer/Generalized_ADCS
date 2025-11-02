__all__ = ["Dipole_Disturbance"]

import numpy as np
from typing import TYPE_CHECKING
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.actuators.noise import Noise
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class Dipole_Disturbance(Disturbance):
    r"""
    **Magnetic Dipole Disturbance Model**

    This class models the **disturbance torque** caused by a residual or parasitic
    magnetic dipole moment interacting with the Earth's geomagnetic field.
    It represents a **passive disturbance torque** that acts on the spacecraft body
    independently of control actuation.

    **Physical Model**

    The disturbance torque due to the interaction between a magnetic dipole
    :math:`\mathbf{m}_d` and the magnetic field :math:`\mathbf{B}_b` is given by:

    .. math::

        \mathbf{T}_d = \mathbf{m}_d \times \mathbf{B}_b

    where:

    - :math:`\mathbf{m}_d` — Residual (or parasitic) magnetic dipole vector [A·m²]
    - :math:`\mathbf{B}_b` — Geomagnetic field vector in the body frame [T]

    The model optionally adds stochastic noise to the dipole vector at each update step
    to represent fluctuations or modeling uncertainty.

    Parameters
    ----------
    dipole_torque : :class:`numpy.ndarray`
        Nominal disturbance dipole vector [A·m²], shape ``(3,)``.

    noise : :class:`~ADCS.satellite_hardware.actuators.noise.Noise`
        Noise model instance that injects random perturbations into the dipole vector.
    """

    def __init__(self, dipole_torque: np.ndarray, noise: Noise):
        r"""
        Initialize the dipole disturbance model.

        Parameters
        ----------
        dipole_torque : :class:`numpy.ndarray`
            Nominal dipole vector (3,) representing the spacecraft’s residual magnetic dipole [A·m²].

        noise : :class:`~ADCS.satellite_hardware.actuators.noise.Noise`
            Instance of the noise model to generate random variations in dipole strength.
        """
        self.torque_nominal = dipole_torque
        self.noise = noise
        self.current_torque = self.torque_nominal.copy()

    def update(self) -> None:
        r"""
        Update the current disturbance torque vector.

        Adds stochastic noise to the nominal dipole vector to simulate time-varying
        disturbance characteristics:

        .. math::

            \mathbf{m}_d(t) = \mathbf{m}_{d,0} + \mathbf{n}(t)

        where:

        - :math:`\mathbf{m}_{d,0}` — Nominal dipole vector  
        - :math:`\mathbf{n}(t)` — Noise realization at time *t*

        This method should be called once per simulation step before computing torque.
        """
        self.current_torque = self.torque_nominal + self.noise.get_noise()

    def torque(self, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **magnetic disturbance torque**.

        The torque is given by:

        .. math::

            \mathbf{T}_d = \mathbf{m}_d \times \mathbf{B}_b

        where the magnetic field :math:`\mathbf{B}_b` is obtained from the current
        :class:`~ADCS.orbits.orbital_state.Orbital_State`.

        Parameters
        ----------
        q : :class:`numpy.ndarray`
            Satellite attitude quaternion (4,).

        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state instance providing the local magnetic field vector.

        Returns
        -------
        :class:`numpy.ndarray`
            Disturbance torque vector in body frame [N·m], shape ``(3,)``.
        """
        vecs = os.get_state_vector(q0=q)
        B_B = vecs["b"]
        return np.cross(self.current_torque, B_B)

    def torque_qjac(self, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **Jacobian of the disturbance torque with respect to the attitude quaternion**.

        Using:

        .. math::

            \frac{\partial \mathbf{T}_d}{\partial q}
            = \mathbf{m}_d \times \frac{\partial \mathbf{B}_b}{\partial q}

        Parameters
        ----------
        q : :class:`numpy.ndarray`
            Satellite attitude quaternion (4,).

        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state instance providing magnetic field derivatives ``∂B_b/∂q``.

        Returns
        -------
        :class:`numpy.ndarray`
            Jacobian matrix ``∂T_d/∂q`` of shape ``(3, 4)``.
        """
        vecs = os.get_state_vector(q0=q)
        db_body__dq = vecs["db"]
        return np.cross(self.current_torque, db_body__dq)

    def torque_qqhess(self, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **second-order derivative (Hessian)** of the disturbance torque with respect to the quaternion.

        The second derivative is:

        .. math::

            \frac{\partial^2 \mathbf{T}_d}{\partial q^2}
            = \mathbf{m}_d \times \frac{\partial^2 \mathbf{B}_b}{\partial q^2}

        Parameters
        ----------
        q : :class:`numpy.ndarray`
            Satellite attitude quaternion (4,).

        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Provides higher-order magnetic field derivatives ``∂²B_b/∂q²``.

        Returns
        -------
        :class:`numpy.ndarray`
            Hessian tensor ``∂²T_d/∂q²`` of shape ``(3, 4, 4)``.
        """
        vecs = os.get_state_vector(q0=q)
        ddb_body__dqdq = vecs["ddb"]
        return np.cross(self.current_torque, ddb_body__dqdq)

    def torque_valjac(self, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **Jacobian of the torque with respect to the dipole value**.

        Since :math:`\mathbf{T}_d = \mathbf{m}_d \times \mathbf{B}_b`,  
        the derivative w.r.t. :math:`\mathbf{m}_d` is:

        .. math::

            \frac{\partial \mathbf{T}_d}{\partial \mathbf{m}_d}
            = [\mathbf{B}_b \times]

        where ``[·×]`` denotes the skew-symmetric cross-product matrix.

        Parameters
        ----------
        q : :class:`numpy.ndarray`
            Satellite attitude quaternion (4,).

        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Orbital state instance providing magnetic field ``\mathbf{B}_b``.

        Returns
        -------
        :class:`numpy.ndarray`
            Jacobian ``∂T_d/∂m_d`` of shape ``(3, 3)``.
        """
        vecs = os.get_state_vector(q0=q)
        B_B = vecs["b"]
        return np.cross(np.eye(3), B_B)

    def torque_qvalhess(self, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Compute the **mixed second derivative** of torque with respect to quaternion and dipole vector.

        .. math::

            \frac{\partial^2 \mathbf{T}_d}{\partial q \, \partial \mathbf{m}_d}
            = \frac{\partial}{\partial q} \left( [\mathbf{B}_b \times] \right)
            = [\frac{\partial \mathbf{B}_b}{\partial q} \times]

        Parameters
        ----------
        q : :class:`numpy.ndarray`
            Satellite attitude quaternion (4,).

        os : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Provides magnetic field and its quaternion derivative ``∂B_b/∂q``.

        Returns
        -------
        :class:`numpy.ndarray`
            Mixed Hessian ``∂²T_d/∂q∂m_d`` of shape ``(4, 3, 3)``.
        """
        vecs = os.get_state_vector(q0=q)
        db_body__dq = vecs["db"]
        return np.cross(np.expand_dims(np.eye(3), 0), np.expand_dims(db_body__dq, 1))
