__all__ = ["RW"]

import numpy as np
import warnings
from ADCS.satellite_hardware.actuators.actuator import Actuator
from ADCS.satellite_hardware.actuators.bias import Bias
from ADCS.satellite_hardware.actuators.noise import Noise
from ADCS.orbits.orbital_state import Orbital_State

class RW(Actuator):
    r"""
    Reaction Wheel (RW) actuator model for satellite attitude control.

    This class models a single-axis reaction wheel, which produces control torque 
    on the spacecraft body by changing the wheel’s angular momentum.

    The torque applied to the spacecraft by a reaction wheel is given by:

    .. math::

        \mathbf{T}_{\text{RW}} = \mathbf{a}_{\text{axis}} \, u

    where:

    - :math:`\mathbf{a}_{\text{axis}}` is the reaction wheel spin axis (unit vector in body frame),
    - :math:`u` is the commanded torque applied to the wheel motor [N·m].

    The change in wheel momentum over time is:

    .. math::

        \dot{\mathbf{h}} = -\mathbf{T}_{\text{RW}}

    which ensures angular momentum conservation between the wheel and the spacecraft body.

    Optional bias and noise models can be applied to represent systematic offset or random 
    disturbances in the torque command.
    """

    def __init__(
        self,
        axis: np.ndarray,
        max_torque: float,
        J: np.ndarray,
        h: np.ndarray,
        h_max: np.ndarray,
        bias: Bias = None,
        noise: Noise = None,
        estimate_bias: bool = False,
    ) -> None:
        r"""
        Initialize a Reaction Wheel actuator.

        Parameters
        ----------
        axis : np.ndarray
            Unit vector (3,) defining the spin axis of the reaction wheel in the 
            satellite body frame.

        max_torque : float
            Maximum torque the wheel motor can produce [N·m].

        J : np.ndarray
            Wheel moment of inertia [kg·m²].

        h : np.ndarray
            Current angular momentum vector of the wheel [N·m·s].

        h_max : np.ndarray
            Maximum allowable angular momentum (saturation limit) [N·m·s].

        bias : Bias, optional
            Bias model instance representing constant or slowly varying offset 
            in the torque command (default is None).

        noise : Noise, optional
            Noise model instance representing stochastic noise in the torque 
            (default is None).

        estimate_bias : bool, optional
            Whether to include this actuator’s bias in the estimation process 
            (default is False).
        """
        self.J = J
        self.h = h
        self.h_max = h_max
        super().__init__(axis=axis, u_max=max_torque, bias=bias, noise=noise, estimate_bias=estimate_bias)

    def torque(self, command: float, q: np.ndarray, os: Orbital_State, float, bias: bool = False, noise: bool = False) -> float:
        r"""
        Compute the reaction wheel torque applied to the spacecraft body.

        The body torque produced by the wheel is:

        .. math::

            \mathbf{T}_{\text{RW}} = \mathbf{a}_{\text{axis}} (u + b) + \mathbf{n}

        where:

        - :math:`u` is the commanded torque input [N·m],
        - :math:`b` is the actuator bias (if enabled),
        - :math:`\mathbf{n}` is the random noise term (if enabled),
        - :math:`\mathbf{a}_{\text{axis}}` is the wheel axis in body coordinates.

        Parameters
        ----------
        command : float
            Commanded torque applied to the wheel motor [N·m].

        j2000 : float
            Current time in Julian date (J2000) used to update bias evolution.

        bias : bool, optional
            Whether to include bias effects (default is False).

        noise : bool, optional
            Whether to include actuator noise (default is False).

        Returns
        -------
        np.ndarray
            Torque vector applied to the spacecraft body [N·m], shape (3,).
        """
        if abs(command) > self.u_max:
            warnings.warn("requested torque exceeds actuation limit")

        torque = command

        if bias:
            torque += self.bias.get_bias(j2000=os.J2000)

        if noise:
            torque += self.noise.get_noise()

        return torque

    def storage_torque(self, command: float) -> float:
        r"""
        Compute the internal torque applied to the wheel (opposite of body torque).

        By conservation of angular momentum:

        .. math::

            \mathbf{T}_{\text{wheel}} = -\mathbf{T}_{\text{body}} = -\mathbf{a}_{\text{axis}} \, u

        This represents the torque acting **on the wheel itself** (used for momentum storage dynamics).

        Parameters
        ----------
        command : float
            Commanded torque applied to the wheel motor [N·m].

        Returns
        -------
        np.ndarray
            Torque acting on the wheel (negative of body torque), shape (3,).
        """
        if abs(command) > self.u_max:
            warnings.warn("RW Requested Torque exceeds actuation limit")
        return -command

    def update_momentum(self, h: float) -> None:
        if h > self.h_max:
            warnings.warn("RW Angular Momentum exceeds saturation limit")
        self.h = h