__all__ = ["Thruster"]

import numpy as np
import warnings
from typing import Optional

from ADCS.satellite_hardware.actuators.actuator import Actuator
from ADCS.satellite_hardware.actuators.bias import Bias
from ADCS.satellite_hardware.actuators.noise import Noise
from ADCS.satellite_hardware.disturbances.disturbance_mode import DisturbanceMode
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize, skewsym


class Thruster(Actuator):
    r"""
    **Thruster Actuator Model for Attitude Control**

    This class represents a single **thruster** (also called a reaction control system 
    jet or RCS thruster) used for spacecraft attitude control. Thrusters generate torque 
    by expelling mass and creating a force at a moment arm from the spacecraft center of mass.

    Mathematical Model
    ------------------

    **Force Model:**
    
    The thrust force is modeled as proportional to the commanded input :math:`u`:

    .. math::

        \mathbf{F} = \hat{\mathbf{n}} \cdot F_{\max} \cdot u

    where:

    - :math:`\hat{\mathbf{n}}` — Unit vector defining thrust direction in body frame
    - :math:`F_{\max}` — Maximum thrust force [N]
    - :math:`u \in [0, 1]` — Normalized thrust command (0 = off, 1 = full thrust)

    For bipropellant or cold gas thrusters with bidirectional capability, :math:`u \in [-1, 1]`.

    **Torque Model:**

    The torque generated about the spacecraft center of mass is:

    .. math::

        \boldsymbol{\tau} = \mathbf{r} \times \mathbf{F} = \mathbf{r} \times (\hat{\mathbf{n}} \cdot F_{\max} \cdot u)

    where :math:`\mathbf{r}` is the position vector from the spacecraft CoM to the thruster 
    application point in the body frame [m].

    This can be rewritten as:

    .. math::

        \boldsymbol{\tau} = (\mathbf{r} \times \hat{\mathbf{n}}) \cdot F_{\max} \cdot u = \mathbf{a}_{\mathrm{eff}} \cdot u

    where :math:`\mathbf{a}_{\mathrm{eff}} = (\mathbf{r} \times \hat{\mathbf{n}}) \cdot F_{\max}` is the 
    effective torque axis and magnitude for unit command.

    **Propellant Consumption:**

    Mass flow rate follows the rocket equation [1]_:

    .. math::

        \dot{m} = \frac{F}{I_{sp} \cdot g_0}

    where:

    - :math:`I_{sp}` — Specific impulse [s]
    - :math:`g_0 = 9.80665` m/s² — Standard gravity

    **Minimum Impulse Bit (MIB):**

    Real thrusters have a minimum on-time, leading to a minimum impulse bit [2]_:

    .. math::

        I_{\min} = F_{\max} \cdot t_{\min}

    This is modeled via the ``min_on_time`` parameter.

    **Thruster Types Supported:**

    1. **Cold Gas** — Simple, low Isp (~50-80 s), commonly used on CubeSats [3]_
    2. **Monopropellant** — Hydrazine or "green" propellants, Isp ~200-230 s [4]_
    3. **Bipropellant** — MMH/NTO or similar, Isp ~290-320 s [5]_
    4. **Electric (Pulsed)** — Can be modeled with appropriate Isp and thrust

    Parameters
    ----------
    thrust_direction : np.ndarray
        Unit vector (3,) defining thrust direction in body frame.
        Will be normalized internally.
    
    position : np.ndarray
        Position vector (3,) from spacecraft CoM to thruster [m].
    
    max_thrust : float
        Maximum thrust force [N].
    
    isp : float
        Specific impulse [s]. Typical values:
        - Cold gas: 50-80 s
        - Monopropellant (hydrazine): 220-230 s
        - Bipropellant: 290-320 s
    
    min_on_time : float, optional
        Minimum thruster on-time [s], determines minimum impulse bit.
        Default is 0.0 (ideal thruster).
    
    bidirectional : bool, optional
        If True, thruster can fire in both directions (u in [-1, 1]).
        If False, u in [0, 1] only. Default is False.
    
    bias : Bias, optional
        Bias model for thrust uncertainty.
    
    noise : Noise, optional
        Noise model for stochastic thrust variations.
    
    estimate_bias : bool, optional
        Whether to include bias in state estimation.

    Attributes
    ----------
    position : np.ndarray
        Thruster position relative to CoM [m].
    
    thrust_direction : np.ndarray
        Normalized thrust direction in body frame.
    
    max_thrust : float
        Maximum thrust [N].
    
    isp : float
        Specific impulse [s].
    
    min_on_time : float
        Minimum on-time [s].
    
    bidirectional : bool
        Whether bidirectional firing is allowed.
    
    effective_torque_axis : np.ndarray
        The direction and magnitude of torque per unit command:
        :math:`\mathbf{a}_{\mathrm{eff}} = (\mathbf{r} \times \hat{\mathbf{n}}) \cdot F_{\max}`
    
    total_impulse : float
        Accumulated impulse [N·s] (for propellant tracking).
    
    total_mass_expended : float
        Accumulated mass expelled [kg].

    References
    ----------
    .. [1] Sutton, G. P., & Biblarz, O. (2016). Rocket Propulsion Elements (9th ed.). 
           Wiley. Chapter 2: Definitions and Fundamentals.
    
    .. [2] Wertz, J. R., & Larson, W. J. (1999). Space Mission Analysis and Design 
           (3rd ed.). Microcosm Press. Section 17.2: Reaction Control Systems.
    
    .. [3] Lemmer, K. (2017). Propulsion for CubeSats. Acta Astronautica, 134, 231-243.
           https://doi.org/10.1016/j.actaastro.2017.01.048
    
    .. [4] Anflo, K., et al. (2008). Flight demonstration of new thruster and green 
           propellant technology on the PRISMA satellite. Acta Astronautica, 65(9-10), 
           1238-1249. https://doi.org/10.1016/j.actaastro.2009.03.056
    
    .. [5] Humble, R. W., Henry, G. N., & Larson, W. J. (1995). Space Propulsion 
           Analysis and Design. McGraw-Hill. Chapter 5: Liquid Rocket Engines.

    Examples
    --------
    Create a cold gas thruster for CubeSat attitude control:

    >>> import numpy as np
    >>> from ADCS.satellite_hardware.actuators import Thruster
    >>> 
    >>> # Thruster mounted at corner, firing in +X direction
    >>> thruster = Thruster(
    ...     thrust_direction=np.array([1, 0, 0]),
    ...     position=np.array([0.05, 0.05, 0.05]),  # 5cm from CoM
    ...     max_thrust=0.1,  # 100 mN cold gas thruster
    ...     isp=65,          # Cold gas specific impulse
    ...     min_on_time=0.01  # 10 ms minimum pulse
    ... )
    >>> 
    >>> # Effective torque axis
    >>> print(f"Torque axis: {thruster.effective_torque_axis}")

    Create a pair of bipropellant thrusters for pitch control:

    >>> thruster_pos = Thruster(
    ...     thrust_direction=np.array([0, 1, 0]),
    ...     position=np.array([1.0, 0, 0]),
    ...     max_thrust=22.0,  # 22 N thruster
    ...     isp=290,
    ...     bidirectional=False
    ... )
    >>> thruster_neg = Thruster(
    ...     thrust_direction=np.array([0, -1, 0]),
    ...     position=np.array([-1.0, 0, 0]),
    ...     max_thrust=22.0,
    ...     isp=290,
    ...     bidirectional=False
    ... )
    """

    # Standard gravity for mass flow calculation
    G0 = 9.80665  # m/s²

    def __init__(
        self,
        thrust_direction: np.ndarray,
        position: np.ndarray,
        max_thrust: float,
        isp: float,
        min_on_time: float = 0.0,
        bidirectional: bool = False,
        bias: Bias = None,
        noise: Noise = None,
        estimate_bias: bool = False,
    ) -> None:
        r"""
        Initialize a thruster actuator.

        Parameters
        ----------
        thrust_direction : np.ndarray
            Thrust direction in body frame (will be normalized).
        
        position : np.ndarray
            Position from CoM to thruster application point [m].
        
        max_thrust : float
            Maximum thrust force [N].
        
        isp : float
            Specific impulse [s].
        
        min_on_time : float, optional
            Minimum on-time [s]. Default 0.0.
        
        bidirectional : bool, optional
            Allow negative commands. Default False.
        
        bias : Bias, optional
            Thrust bias model.
        
        noise : Noise, optional
            Thrust noise model.
        
        estimate_bias : bool, optional
            Include bias in estimation. Default False.
        """
        # Store thruster-specific parameters
        self.thrust_direction = normalize(thrust_direction)
        self.position = np.asarray(position, dtype=float)
        self.max_thrust = float(max_thrust)
        self.isp = float(isp)
        self.min_on_time = float(min_on_time)
        self.bidirectional = bool(bidirectional)

        # Compute effective torque axis: τ = r × (n * F_max * u)
        # For unit command u=1: τ_eff = r × n * F_max
        self.effective_torque_axis = np.cross(self.position, self.thrust_direction) * self.max_thrust

        # Propellant tracking
        self.total_impulse = 0.0  # [N·s]
        self.total_mass_expended = 0.0  # [kg]

        # The actuator "axis" for the parent class is the effective torque direction
        # u_max is 1.0 for normalized command
        if np.linalg.norm(self.effective_torque_axis) > 1e-12:
            axis = self.effective_torque_axis
        else:
            # Thruster aligned with CoM → no torque, but we still need an axis
            axis = self.thrust_direction
        
        super().__init__(
            axis=axis,
            u_max=1.0,  # Normalized command
            bias=bias,
            noise=noise,
            estimate_bias=estimate_bias
        )

    def torque(
        self,
        u: float,
        x: np.ndarray,
        os: Orbital_State,
        dmode: DisturbanceMode = None
    ) -> np.ndarray:
        r"""
        Compute the torque generated by the thruster.

        The torque is:

        .. math::

            \boldsymbol{\tau} = (\mathbf{r} \times \hat{\mathbf{n}}) \cdot F_{\max} \cdot (u + b) + \mathbf{n}

        where :math:`b` is bias and :math:`\mathbf{n}` is noise.

        Parameters
        ----------
        u : float
            Normalized thrust command. Range depends on ``bidirectional``:
            - If False: :math:`u \in [0, 1]`
            - If True: :math:`u \in [-1, 1]`
        
        x : np.ndarray
            Spacecraft state vector (unused for thruster torque, included for API consistency).
        
        os : Orbital_State
            Orbital state (provides J2000 time for bias updates).
        
        dmode : DisturbanceMode, optional
            Controls bias/noise application.

        Returns
        -------
        np.ndarray
            Torque vector in body frame [N·m], shape (3,).

        Warnings
        --------
        Issues UserWarning if command exceeds limits.
        """
        # Validate command
        if not self.bidirectional and u < 0:
            warnings.warn(f"Thruster received negative command u={u} but is not bidirectional")
            u = 0.0
        
        if abs(u) > 1.0:
            warnings.warn(f"Thruster command |u|={abs(u)} exceeds normalized limit of 1.0")

        if dmode is None:
            dmode = DisturbanceMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)

        # Apply bias
        command = u
        if self.bias and dmode.add_bias:
            command += self.bias.get_bias(j2000=os.J2000)
        if dmode.update_bias and self.bias:
            self.bias._update_bias(j2000=os.J2000)

        # Compute torque: τ = (r × n) * F_max * command = effective_torque_axis * command
        torque = self.effective_torque_axis * command

        # Apply noise
        if self.noise and dmode.add_noise:
            # Noise is applied as additive torque perturbation
            torque += self.noise.get_noise() * np.linalg.norm(self.effective_torque_axis)
        if dmode.update_noise and self.noise:
            self.noise._update_noise()

        return torque

    def force(
        self,
        u: float,
        x: np.ndarray,
        os: Orbital_State,
        dmode: DisturbanceMode = None
    ) -> np.ndarray:
        r"""
        Compute the force generated by the thruster.

        .. math::

            \mathbf{F} = \hat{\mathbf{n}} \cdot F_{\max} \cdot (u + b)

        Parameters
        ----------
        u : float
            Normalized thrust command.
        
        x : np.ndarray
            Spacecraft state vector.
        
        os : Orbital_State
            Orbital state.
        
        dmode : DisturbanceMode, optional
            Controls bias/noise application.

        Returns
        -------
        np.ndarray
            Force vector in body frame [N], shape (3,).

        Notes
        -----
        This method is useful for translational dynamics coupling or
        for computing the effect on orbit.
        """
        if not self.bidirectional and u < 0:
            u = 0.0
        
        if dmode is None:
            dmode = DisturbanceMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)

        command = u
        if self.bias and dmode.add_bias:
            command += self.bias.get_bias(j2000=os.J2000)

        force = self.thrust_direction * self.max_thrust * command

        if self.noise and dmode.add_noise:
            force += self.noise.get_noise() * self.max_thrust * self.thrust_direction

        return force

    def mass_flow_rate(self, u: float) -> float:
        r"""
        Compute instantaneous mass flow rate.

        .. math::

            \dot{m} = \frac{F_{\max} \cdot |u|}{I_{sp} \cdot g_0}

        Parameters
        ----------
        u : float
            Normalized thrust command.

        Returns
        -------
        float
            Mass flow rate [kg/s].
        """
        thrust = self.max_thrust * abs(u)
        return thrust / (self.isp * self.G0)

    def update_propellant_usage(self, u: float, dt: float) -> float:
        r"""
        Update propellant tracking and return mass expended.

        Parameters
        ----------
        u : float
            Normalized thrust command.
        
        dt : float
            Time step [s].

        Returns
        -------
        float
            Mass expended in this time step [kg].
        """
        mdot = self.mass_flow_rate(u)
        mass_used = mdot * dt
        impulse = self.max_thrust * abs(u) * dt

        self.total_mass_expended += mass_used
        self.total_impulse += impulse

        return mass_used

    def minimum_impulse_bit(self) -> float:
        r"""
        Return the minimum impulse bit.

        .. math::

            I_{\min} = F_{\max} \cdot t_{\min}

        Returns
        -------
        float
            Minimum impulse bit [N·s].
        """
        return self.max_thrust * self.min_on_time

    def storage_torque(self, u: float, x: np.ndarray, os: Orbital_State, dmode: DisturbanceMode = None) -> np.ndarray:
        r"""
        Thrusters have no momentum storage.

        Returns
        -------
        np.ndarray
            Empty array, shape (0,).
        """
        return np.zeros((0,))

    # ========================================================================
    # JACOBIANS (First Derivatives)
    # ========================================================================

    def dtorq__du(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of torque w.r.t. command: :math:`\partial\boldsymbol{\tau}/\partial u`.

        Since :math:`\boldsymbol{\tau} = \mathbf{a}_{\mathrm{eff}} \cdot u`, we have:

        .. math::

            \frac{\partial\boldsymbol{\tau}}{\partial u} = \mathbf{a}_{\mathrm{eff}}

        Returns
        -------
        np.ndarray
            Shape (1, 3).
        """
        return self.effective_torque_axis.reshape((1, 3))

    def dtorq__dbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of torque w.r.t. state: :math:`\partial\boldsymbol{\tau}/\partial \mathbf{x}`.

        Thruster torque does not depend on spacecraft state (unlike MTQ which depends
        on attitude via B-field), so this is zero.

        Returns
        -------
        np.ndarray
            Shape (7, 3), all zeros.
        """
        return np.zeros((7, 3))

    def dtorq__dh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Derivative w.r.t. momentum storage state.

        Thrusters have no momentum coupling.

        Returns
        -------
        np.ndarray
            Shape (0, 3).
        """
        return np.zeros((0, 3))

    # ========================================================================
    # HESSIANS (Second Derivatives)
    # ========================================================================

    def ddtorq__dudu(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}/\partial u^2`.

        Since torque is linear in u, this is zero.

        Returns
        -------
        np.ndarray
            Shape (1, 1, 3), all zeros.
        """
        return np.zeros((1, 1, 3))

    def ddtorq__dudbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed derivative :math:`\partial^2\boldsymbol{\tau}/\partial u \partial \mathbf{x}`.

        Zero since torque doesn't depend on state.

        Returns
        -------
        np.ndarray
            Shape (1, 7, 3), all zeros.
        """
        return np.zeros((1, 7, 3))

    def ddtorq__dudh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed derivative :math:`\partial^2\boldsymbol{\tau}/\partial u \partial \mathbf{h}`.

        Returns
        -------
        np.ndarray
            Shape (1, 0, 3).
        """
        return np.zeros((1, 0, 3))

    def ddtorq__dbasestatedbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}/\partial \mathbf{x}^2`.

        Returns
        -------
        np.ndarray
            Shape (7, 7, 3), all zeros.
        """
        return np.zeros((7, 7, 3))

    def ddtorq__dbasestatedh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed derivative :math:`\partial^2\boldsymbol{\tau}/\partial \mathbf{x} \partial \mathbf{h}`.

        Returns
        -------
        np.ndarray
            Shape (7, 0, 3).
        """
        return np.zeros((7, 0, 3))

    def ddtorq__dhdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}/\partial \mathbf{h}^2`.

        Returns
        -------
        np.ndarray
            Shape (0, 0, 3).
        """
        return np.zeros((0, 0, 3))

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def reset_propellant_tracking(self) -> None:
        """Reset accumulated propellant usage counters."""
        self.total_impulse = 0.0
        self.total_mass_expended = 0.0

    def __repr__(self) -> str:
        return (
            f"Thruster(dir={self.thrust_direction}, pos={self.position}, "
            f"F_max={self.max_thrust} N, Isp={self.isp} s, "
            f"τ_eff={np.linalg.norm(self.effective_torque_axis):.4f} N·m/cmd)"
        )
