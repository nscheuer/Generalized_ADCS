__all__ = ["General_Disturbance"]

import numpy as np
from typing import Dict, Optional
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.orbits.orbital_state import Orbital_State


class General_Disturbance(Disturbance):
    r"""
    **General (Lumped) Disturbance Model**

    Models an unknown or lumped disturbance torque that can be estimated by
    the attitude filter. This is the "All-in-one" approach where multiple
    unknown disturbances are tracked as a single 3-vector torque.

    **Physical Model**

    The disturbance torque is simply a 3-vector in body frame:

    .. math::

        \mathbf{T}_d = \boldsymbol{\tau}_{gen}

    The estimator treats this as a slowly-varying parameter with random walk
    dynamics:

    .. math::

        \dot{\boldsymbol{\tau}}_{gen} = \mathbf{w}, \quad \mathbf{w} \sim \mathcal{N}(0, Q_{gen})

    Parameters
    ----------
    torque_init : :class:`numpy.ndarray`, optional
        Initial disturbance torque estimate [N·m], shape ``(3,)``. Default zeros.
    std : float or :class:`numpy.ndarray`, optional
        Process noise standard deviation for random walk. If scalar, applied
        to all axes. Default 1e-6 N·m/√s.
    mag_max : float, optional
        Maximum magnitude limit for the disturbance [N·m]. Default 1e-3.
    estimate_dist : bool, optional
        If True, this disturbance is included in the estimator state. Default True.

    Attributes
    ----------
    main_param : :class:`numpy.ndarray`
        Current disturbance torque estimate [N·m], shape ``(3,)``.
    std : :class:`numpy.ndarray`
        Process noise covariance matrix, shape ``(3,3)``.
    mag_max : float
        Maximum allowed torque magnitude.
    active : bool
        If False, torque returns zero (inherited from Disturbance).

    Notes
    -----
    This disturbance model is particularly useful when:
    
    1. The true disturbance sources are unknown or too complex to model
    2. Multiple small disturbances can be lumped together
    3. A simple, robust estimation approach is preferred over detailed modeling

    The estimator will track this torque as part of the augmented state vector,
    allowing the controller to compensate for it via feedforward.

    Example
    -------
    >>> gen_dist = General_Disturbance(
    ...     torque_init=np.zeros(3),
    ...     std=5e-6,
    ...     mag_max=1e-3,
    ...     estimate_dist=True
    ... )
    >>> sat = Satellite(..., disturbances=[gen_dist])
    """

    def __init__(
        self,
        torque_init: np.ndarray = None,
        std: float | np.ndarray = 1e-6,
        mag_max: float = 1e-3,
        estimate_dist: bool = True,
    ):
        super().__init__(estimate_dist=estimate_dist, estimated_vector_length=3)
        
        # Initialize torque estimate
        if torque_init is None:
            self.main_param = np.zeros(3)
        else:
            self.main_param = np.asarray(torque_init, dtype=float).reshape(3)
        
        # Process noise covariance
        if np.isscalar(std):
            self.std = np.eye(3) * std
        else:
            self.std = np.asarray(std, dtype=float)
            if self.std.shape == (3,):
                self.std = np.diag(self.std)
        
        self.mag_max = float(mag_max)
        self.active = True
        self.last_update_time = np.nan

    def torque(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Returns the current disturbance torque estimate.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            State vector (unused, for interface compatibility).
        os : Orbital_State
            Current orbital state (unused, for interface compatibility).

        Returns
        -------
        :class:`numpy.ndarray`
            Disturbance torque in body frame [N·m], shape ``(3,)``.
        """
        if not self.active:
            return np.zeros(3)
        return self.main_param.copy()

    def torque_valjac(self, sat, vecs: Dict[str, np.ndarray]) -> np.ndarray:
        r"""
        Jacobian of torque with respect to the disturbance parameter.

        Since torque = main_param directly, the Jacobian is identity.

        Parameters
        ----------
        sat : Satellite
            Satellite object.
        vecs : dict
            Environmental vectors (unused).

        Returns
        -------
        :class:`numpy.ndarray`
            Jacobian matrix, shape ``(3, 3)`` = Identity.
        """
        if not self.active:
            return np.zeros((3, 3))
        return np.eye(3)

    def torque_valvalhess(self, sat, vecs: Dict[str, np.ndarray]) -> np.ndarray:
        r"""
        Hessian of torque with respect to the disturbance parameter.

        Since torque is linear in main_param, Hessian is zero.

        Returns
        -------
        :class:`numpy.ndarray`
            Hessian tensor, shape ``(3, 3, 3)`` = zeros.
        """
        return np.zeros((3, 3, 3))

    def torque_qjac(self, sat, vecs: Dict[str, np.ndarray]) -> np.ndarray:
        r"""
        Jacobian of torque with respect to quaternion.

        General disturbance is attitude-independent.

        Returns
        -------
        :class:`numpy.ndarray`
            Jacobian, shape ``(4, 3)`` = zeros.
        """
        return np.zeros((4, 3))

    def torque_qqhess(self, sat, vecs: Dict[str, np.ndarray]) -> np.ndarray:
        r"""
        Hessian of torque with respect to quaternion.

        Returns
        -------
        :class:`numpy.ndarray`
            Hessian tensor, shape ``(4, 4, 3)`` = zeros.
        """
        return np.zeros((4, 4, 3))

    def torque_qvalhess(self, sat, vecs: Dict[str, np.ndarray]) -> np.ndarray:
        r"""
        Mixed Hessian of torque with respect to quaternion and parameter.

        Returns
        -------
        :class:`numpy.ndarray`
            Hessian tensor, shape ``(4, 3, 3)`` = zeros.
        """
        return np.zeros((4, 3, 3))

    def update(self, J2000: float) -> "General_Disturbance":
        r"""
        Update the disturbance estimate with random walk dynamics.

        This is called by the estimator's propagation step to evolve
        the disturbance estimate according to:

        .. math::

            \boldsymbol{\tau}_{gen}(t+dt) = \boldsymbol{\tau}_{gen}(t) + \mathbf{w} \cdot dt

        with magnitude limiting applied afterward.

        Parameters
        ----------
        J2000 : float
            Current time in Julian centuries since J2000.

        Returns
        -------
        General_Disturbance
            Self, for method chaining.
        """
        from ADCS.orbits.universal_constants import TimeConstants
        
        if np.isnan(self.last_update_time):
            self.last_update_time = J2000
            return self
        
        if J2000 > self.last_update_time:
            dt_sec = (J2000 - self.last_update_time) * TimeConstants.cent2sec
            
            # Random walk update: covariance should scale linearly with dt
            # For Wiener process: cov = dt * Q where Q = self.std @ self.std.T
            noise = np.random.multivariate_normal(
                np.zeros(3),
                dt_sec * (self.std @ self.std.T)
            )
            new_torque = self.main_param + noise
            
            # Apply magnitude limit
            mag = np.linalg.norm(new_torque)
            if mag > self.mag_max:
                new_torque = new_torque * (self.mag_max / mag)
            
            self.main_param = new_torque
            self.last_update_time = J2000
        
        return self

    def set_estimate(self, value: np.ndarray) -> None:
        r"""
        Set the disturbance torque estimate directly.

        Used by the estimator to update the disturbance after a measurement update.

        Parameters
        ----------
        value : :class:`numpy.ndarray`
            New disturbance torque estimate [N·m], shape ``(3,)``.
        """
        self.main_param = np.asarray(value, dtype=float).reshape(3)
        
        # Apply magnitude limit
        mag = np.linalg.norm(self.main_param)
        if mag > self.mag_max:
            self.main_param = self.main_param * (self.mag_max / mag)

    def get_estimate(self) -> np.ndarray:
        r"""
        Get the current disturbance torque estimate.

        Returns
        -------
        :class:`numpy.ndarray`
            Current disturbance torque estimate [N·m], shape ``(3,)``.
        """
        return self.main_param.copy()
