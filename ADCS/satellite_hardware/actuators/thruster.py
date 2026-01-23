__all__ = ["Thruster"]

import numpy as np
import warnings
from typing import Optional, Tuple
from enum import Enum

from ADCS.satellite_hardware.actuators.actuator import Actuator
from ADCS.satellite_hardware.actuators.bias import Bias
from ADCS.satellite_hardware.actuators.noise import Noise
from ADCS.satellite_hardware.disturbances.disturbance_mode import DisturbanceMode
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize, skewsym


class MIBBehavior(Enum):
    """
    Behavior when commanded thrust is below minimum impulse bit.
    
    Physical thrusters have a minimum on-time due to valve dynamics.
    Commands below this threshold must be handled:
    
    - QUANTIZE_TO_ZERO: Command below MIB produces zero thrust (conservative)
    - QUANTIZE_TO_MIB: Command below MIB produces full MIB pulse (wastes propellant)
    - ACCUMULATE: Accumulate small commands until MIB is reached (requires state)
    
    References
    ----------
    .. [1] Wie, B. (2008). Space Vehicle Dynamics and Control (2nd ed.). 
           AIAA. Section 7.4.2: Thruster Minimum Impulse Bit Effects.
    """
    QUANTIZE_TO_ZERO = "zero"
    QUANTIZE_TO_MIB = "mib"
    ACCUMULATE = "accumulate"


# Global flag to track if thruster integration warning has been shown
_THRUSTER_INTEGRATION_WARNING_SHOWN = False


class Thruster(Actuator):
    r"""
    **Thruster Actuator Model for Attitude Control**

    .. warning::
        **INTEGRATION STATUS: EXPERIMENTAL**
        
        Thruster integration with the control allocation system is not yet complete.
        The following features need implementation before operational use:
        
        1. Control allocation (LP/QP) does not yet support thrusters
        2. Fuel consumption is tracked but not integrated with mission planning
        3. Minimum impulse bit effects on closed-loop stability need analysis
        4. Force effects on translational dynamics are not propagated
        
        Use with caution and verify behavior for your specific application.

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

    **Minimum Impulse Bit (MIB):**

    Real thrusters have a minimum on-time due to valve actuation dynamics [1]_:

    .. math::

        I_{\min} = F_{\max} \cdot t_{\min}

    Commands below this threshold are handled according to ``mib_behavior``:
    
    - ``QUANTIZE_TO_ZERO``: No thrust (conservative, may cause limit cycles)
    - ``QUANTIZE_TO_MIB``: Full MIB pulse (wastes propellant, better tracking)
    - ``ACCUMULATE``: Integrate small commands until MIB reached (complex)

    **Propellant Consumption:**

    Mass flow rate follows the rocket equation [2]_:

    .. math::

        \dot{m} = \frac{F}{I_{sp} \cdot g_0}

    Parameters
    ----------
    thrust_direction : np.ndarray
        Unit vector (3,) defining thrust direction in body frame.
    
    position : np.ndarray
        Position vector (3,) from spacecraft CoM to thruster [m].
    
    max_thrust : float
        Maximum thrust force [N].
    
    isp : float
        Specific impulse [s].
    
    min_on_time : float, optional
        Minimum thruster on-time [s]. Default 0.0 (ideal thruster).
    
    mib_behavior : MIBBehavior, optional
        How to handle commands below MIB. Default QUANTIZE_TO_ZERO.
    
    bidirectional : bool, optional
        If True, thruster can fire in both directions. Default False.
    
    control_dt : float, optional
        Control loop timestep [s]. Used to convert normalized command to impulse.
        Default 1.0.
    
    bias : Bias, optional
        Bias model for thrust uncertainty.
    
    noise : Noise, optional
        Noise model for stochastic thrust variations.

    Attributes
    ----------
    effective_torque_axis : np.ndarray
        Torque per unit command: :math:`(\mathbf{r} \times \hat{\mathbf{n}}) \cdot F_{\max}`
    
    total_impulse : float
        Accumulated impulse [N·s].
    
    total_mass_expended : float
        Accumulated mass expelled [kg].
    
    accumulated_command : float
        For ACCUMULATE mode: integrated sub-MIB commands.
    
    firing_count : int
        Number of thruster firings (for lifetime tracking).

    References
    ----------
    .. [1] Wie, B. (2008). Space Vehicle Dynamics and Control (2nd ed.). 
           AIAA. Section 7.4.2: Thruster Minimum Impulse Bit Effects.
    
    .. [2] Sutton, G. P., & Biblarz, O. (2016). Rocket Propulsion Elements (9th ed.). 
           Wiley. Chapter 2: Definitions and Fundamentals.
    
    .. [3] Wertz, J. R., & Larson, W. J. (1999). Space Mission Analysis and Design 
           (3rd ed.). Microcosm Press. Section 17.2: Reaction Control Systems.
    
    .. [4] Lemmer, K. (2017). Propulsion for CubeSats. Acta Astronautica, 134, 231-243.
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
        mib_behavior: MIBBehavior = MIBBehavior.QUANTIZE_TO_ZERO,
        bidirectional: bool = False,
        control_dt: float = 1.0,
        bias: Bias = None,
        noise: Noise = None,
        estimate_bias: bool = False,
    ) -> None:
        # Store thruster-specific parameters
        self.thrust_direction = normalize(thrust_direction)
        self.position = np.asarray(position, dtype=float)
        self.max_thrust = float(max_thrust)
        self.isp = float(isp)
        self.min_on_time = float(min_on_time)
        self.mib_behavior = mib_behavior
        self.bidirectional = bool(bidirectional)
        self.control_dt = float(control_dt)

        # Compute effective torque axis: τ = r × (n * F_max * u)
        self.effective_torque_axis = np.cross(self.position, self.thrust_direction) * self.max_thrust

        # Propellant and lifetime tracking
        self.total_impulse = 0.0  # [N·s]
        self.total_mass_expended = 0.0  # [kg]
        self.accumulated_command = 0.0  # For ACCUMULATE mode
        self.firing_count = 0  # Number of firings

        # Compute minimum normalized command that produces MIB
        # MIB = F_max * t_min, so u_min = t_min / control_dt (for one control step)
        if self.control_dt > 0 and self.min_on_time > 0:
            self.u_min = self.min_on_time / self.control_dt
        else:
            self.u_min = 0.0

        # The actuator "axis" for the parent class
        if np.linalg.norm(self.effective_torque_axis) > 1e-12:
            axis = self.effective_torque_axis
        else:
            axis = self.thrust_direction
        
        super().__init__(
            axis=axis,
            u_max=1.0,
            bias=bias,
            noise=noise,
            estimate_bias=estimate_bias
        )

    def _issue_integration_warning(self) -> None:
        """Issue one-time warning about thruster integration status."""
        global _THRUSTER_INTEGRATION_WARNING_SHOWN
        if not _THRUSTER_INTEGRATION_WARNING_SHOWN:
            warnings.warn(
                "\n" + "="*70 + "\n"
                "THRUSTER INTEGRATION WARNING\n"
                "="*70 + "\n"
                "Thruster actuators are EXPERIMENTAL. Key limitations:\n"
                "  1. Control allocation (LP/QP) does not yet support thrusters\n"
                "  2. Fuel consumption not integrated with mission planning\n"
                "  3. MIB effects on closed-loop stability not fully analyzed\n"
                "  4. Translational dynamics coupling not implemented\n"
                "\n"
                "This warning appears once per session. Verify thruster behavior\n"
                "carefully for your specific application.\n"
                "="*70,
                category=UserWarning,
                stacklevel=3
            )
            _THRUSTER_INTEGRATION_WARNING_SHOWN = True

    def _apply_mib_quantization(self, u: float) -> Tuple[float, bool]:
        """
        Apply minimum impulse bit quantization to command.
        
        Parameters
        ----------
        u : float
            Raw normalized command.
        
        Returns
        -------
        u_quantized : float
            Quantized command after MIB logic.
        fired : bool
            Whether the thruster actually fired.
        """
        u_abs = abs(u)
        sign = np.sign(u) if u != 0 else 1.0
        
        # No MIB constraint
        if self.u_min <= 0:
            return u, (u_abs > 1e-10)
        
        # Command above MIB threshold - fire normally
        if u_abs >= self.u_min:
            return u, True
        
        # Command below MIB threshold
        if self.mib_behavior == MIBBehavior.QUANTIZE_TO_ZERO:
            # Don't fire - command too small
            if u_abs > 1e-10:
                warnings.warn(
                    f"Thruster command |u|={u_abs:.4f} below MIB threshold u_min={self.u_min:.4f}. "
                    f"Quantized to ZERO (no firing). Consider adjusting control gains or MIB behavior.",
                    category=UserWarning,
                    stacklevel=4
                )
            return 0.0, False
        
        elif self.mib_behavior == MIBBehavior.QUANTIZE_TO_MIB:
            # Fire full MIB pulse
            if u_abs > 1e-10:
                warnings.warn(
                    f"Thruster command |u|={u_abs:.4f} below MIB threshold u_min={self.u_min:.4f}. "
                    f"Quantized to MIB={self.u_min:.4f} (may waste propellant).",
                    category=UserWarning,
                    stacklevel=4
                )
            return sign * self.u_min, True
        
        elif self.mib_behavior == MIBBehavior.ACCUMULATE:
            # Accumulate until MIB reached
            self.accumulated_command += u_abs
            
            if self.accumulated_command >= self.u_min:
                # Fire accumulated pulse
                fire_amount = sign * self.accumulated_command
                self.accumulated_command = 0.0
                warnings.warn(
                    f"Thruster accumulated command reached MIB. Firing pulse u={fire_amount:.4f}.",
                    category=UserWarning,
                    stacklevel=4
                )
                return fire_amount, True
            else:
                # Keep accumulating
                return 0.0, False
        
        return u, (u_abs > 1e-10)

    def torque(
        self,
        u: float,
        x: np.ndarray,
        os: Orbital_State,
        dmode: DisturbanceMode = None
    ) -> np.ndarray:
        r"""
        Compute the torque generated by the thruster.

        .. warning::
            This method includes MIB quantization. The actual torque produced
            may differ from a linear model due to thruster physics.

        Parameters
        ----------
        u : float
            Normalized thrust command in [0, 1] or [-1, 1] if bidirectional.
        
        x : np.ndarray
            Spacecraft state vector.
        
        os : Orbital_State
            Orbital state.
        
        dmode : DisturbanceMode, optional
            Controls bias/noise application.

        Returns
        -------
        np.ndarray
            Torque vector in body frame [N·m], shape (3,).
        """
        # Issue integration warning on first use
        self._issue_integration_warning()
        
        # Validate command bounds
        if not self.bidirectional and u < 0:
            warnings.warn(
                f"Thruster received negative command u={u:.4f} but is not bidirectional. "
                f"Clamping to zero.",
                category=UserWarning,
                stacklevel=2
            )
            u = 0.0
        
        if abs(u) > 1.0:
            warnings.warn(
                f"Thruster command |u|={abs(u):.4f} exceeds normalized limit of 1.0. "
                f"Clamping to ±1.0.",
                category=UserWarning,
                stacklevel=2
            )
            u = np.clip(u, -1.0 if self.bidirectional else 0.0, 1.0)

        if dmode is None:
            dmode = DisturbanceMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)

        # Apply bias before MIB quantization
        command = u
        if self.bias and dmode.add_bias:
            command += self.bias.get_bias(j2000=os.J2000)
        if dmode.update_bias and self.bias:
            self.bias._update_bias(j2000=os.J2000)

        # Apply MIB quantization
        command_quantized, fired = self._apply_mib_quantization(command)
        
        # Track firing
        if fired:
            self.firing_count += 1

        # Compute torque
        torque = self.effective_torque_axis * command_quantized

        # Apply noise
        if self.noise and dmode.add_noise and fired:
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

        .. warning::
            Force is computed but NOT currently propagated to translational dynamics.
            This method is for analysis/logging purposes only.

        Parameters
        ----------
        u : float
            Normalized thrust command.
        
        Returns
        -------
        np.ndarray
            Force vector in body frame [N], shape (3,).
        """
        if not self.bidirectional and u < 0:
            u = 0.0
        
        if dmode is None:
            dmode = DisturbanceMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)

        command = u
        if self.bias and dmode.add_bias:
            command += self.bias.get_bias(j2000=os.J2000)

        # Apply MIB quantization
        command_quantized, _ = self._apply_mib_quantization(command)

        force = self.thrust_direction * self.max_thrust * command_quantized

        if self.noise and dmode.add_noise:
            force += self.noise.get_noise() * self.max_thrust * self.thrust_direction

        return force

    def mass_flow_rate(self, u: float) -> float:
        r"""
        Compute instantaneous mass flow rate.

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

        .. note::
            This must be called explicitly by the simulation loop.
            Automatic propellant tracking in torque() is not implemented
            to avoid double-counting in multi-rate simulations.

        Parameters
        ----------
        u : float
            Normalized thrust command (should be post-MIB-quantization).
        
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
        """Return the minimum impulse bit [N·s]."""
        return self.max_thrust * self.min_on_time

    def storage_torque(self, u: float, x: np.ndarray, os: Orbital_State, 
                       dmode: DisturbanceMode = None) -> np.ndarray:
        """Thrusters have no momentum storage. Returns empty array."""
        return np.zeros((0,))

    # ========================================================================
    # JACOBIANS (First Derivatives)
    # ========================================================================
    # Note: Jacobians assume linear model (no MIB quantization).
    # This is appropriate for trajectory optimization but may not reflect
    # actual closed-loop behavior with MIB effects.

    def dtorq__du(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative :math:`\partial\boldsymbol{\tau}/\partial u`.

        .. note::
            Returns linear derivative. Does not account for MIB quantization
            discontinuities (appropriate for trajectory optimization).
        """
        return self.effective_torque_axis.reshape((1, 3))

    def dtorq__dbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        """Thruster torque is independent of spacecraft state."""
        return np.zeros((7, 3))

    def dtorq__dh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        """Thrusters have no momentum coupling."""
        return np.zeros((0, 3))

    # ========================================================================
    # HESSIANS (Second Derivatives) - All zero for linear torque model
    # ========================================================================

    def ddtorq__dudu(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((1, 1, 3))

    def ddtorq__dudbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((1, 7, 3))

    def ddtorq__dudh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((1, 0, 3))

    def ddtorq__dbasestatedbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((7, 7, 3))

    def ddtorq__dbasestatedh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((7, 0, 3))

    def ddtorq__dhdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((0, 0, 3))

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def reset_propellant_tracking(self) -> None:
        """Reset accumulated propellant usage counters."""
        self.total_impulse = 0.0
        self.total_mass_expended = 0.0
        self.accumulated_command = 0.0
        self.firing_count = 0

    def get_status(self) -> dict:
        """
        Get thruster status for logging/telemetry.
        
        Returns
        -------
        dict
            Status information including propellant usage and firing count.
        """
        return {
            'total_impulse_Ns': self.total_impulse,
            'total_mass_kg': self.total_mass_expended,
            'firing_count': self.firing_count,
            'accumulated_command': self.accumulated_command,
            'mib_behavior': self.mib_behavior.value,
        }

    def __repr__(self) -> str:
        return (
            f"Thruster(dir={self.thrust_direction}, pos={self.position}, "
            f"F_max={self.max_thrust} N, Isp={self.isp} s, "
            f"MIB={self.minimum_impulse_bit():.4f} N·s, "
            f"τ_eff={np.linalg.norm(self.effective_torque_axis):.4f} N·m/cmd)"
        )


def reset_thruster_warnings():
    """Reset the global thruster integration warning flag (for testing)."""
    global _THRUSTER_INTEGRATION_WARNING_SHOWN
    _THRUSTER_INTEGRATION_WARNING_SHOWN = False
