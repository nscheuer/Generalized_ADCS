__all__ = ["Actuator"]

import numpy as np
from ADCS.satellite_hardware.errors.bias import Bias
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.helpers.math_helpers import normalize

class Actuator:
    r"""
    Abstract base class for all spacecraft actuators in the ADCS framework.

    This class defines a **common mathematical and software interface** for
    torque–producing devices such as magnetorquers, thrusters, or reaction wheels.
    It provides a complete set of first- and second-order derivatives required
    by nonlinear estimation and control algorithms (e.g., EKF, iLQR, DDP),
    while leaving the physical torque model to subclasses.

    --------------------
    Mathematical model
    --------------------

    An actuator produces a **body-frame torque**

    .. math::

        \boldsymbol{\tau}
        = \boldsymbol{\tau}\!\left(
            u,\,
            \mathbf{x},\,
            \mathrm{os}
          \right)
        \in \mathbb{R}^3

    where

    .. math::

        \mathbf{x}
        =
        \begin{bmatrix}
            \boldsymbol{\omega} \\
            \mathbf{q}
        \end{bmatrix}
        \in \mathbb{R}^7

    is the base spacecraft state consisting of angular velocity
    :math:`\boldsymbol{\omega}\in\mathbb{R}^3` and attitude quaternion
    :math:`\mathbf{q}\in\mathbb{R}^4`.

    The actuator command is a **scalar input**

    .. math::

        u \in [-u_{\max},\, u_{\max}]

    applied along a **fixed body-frame axis**

    .. math::

        \mathbf{a} = \frac{\texttt{axis}}{\|\texttt{axis}\|}
        \in \mathbb{S}^2

    Bias and noise models are optional and are represented by
    :class:`~ADCS.satellite_hardware.errors.bias.Bias`
    and
    :class:`~ADCS.satellite_hardware.errors.noise.Noise`.

    --------------------
    Design philosophy
    --------------------

    * All Jacobians are **row-Jacobians** (derivative of a vector w.r.t. a scalar
      or vector).
    * All second derivatives return **rank-3 tensors** with the final dimension
      corresponding to the torque vector.
    * Empty dimensions (size 0) are used when a state does not exist.
    * This base class returns zeros everywhere; subclasses must override
      the relevant methods.

    See also:
        :class:`~ADCS.satellite_hardware.actuators.magnetotorquer.MTQ`,
        :class:`~ADCS.satellite_hardware.actuators.reaction_wheel.RW`
    """
    def __init__(self, axis: np.ndarray, u_max: float, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False) -> None:
        r"""
        Initialize an actuator model.

        The actuation axis is normalized internally and the command magnitude
        is bounded by ``u_max``.

        :param axis: Actuation direction in the body frame.
        :type axis: numpy.ndarray of shape ``(3,)``

        :param u_max: Maximum admissible command magnitude :math:`u_{\max}`.
        :type u_max: float

        :param bias: Optional actuator bias model.
        :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` or None

        :param noise: Optional actuator noise model.
        :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` or None

        :param estimate_bias: Flag indicating whether bias estimation is enabled.
        :type estimate_bias: bool

        :return: None
        :rtype: None
        """
        self.axis = normalize(axis)
        self.u_max = u_max
        if bias:
            self.bias: Bias = bias.copy()
        else:
            self.bias = Bias()
        if noise:
            self.noise: Noise = noise.copy()
        else:
            self.noise = Noise()
        self.estimate_bias: bool = estimate_bias
        self.input_len: int = 1
        self.last_bias_time: float = float('nan')

    def estimate_torque_capability(
        self, 
        orbital_states: list = None,
        n_samples: int = 100,
        expected_field_uT: float = 30.0
    ) -> float:
        r"""
        Estimate the maximum torque this actuator can produce.
        
        For environment-independent actuators (e.g., reaction wheels, thrusters),
        this returns ``u_max`` directly since torque = u_max * axis.
        
        For environment-dependent actuators (e.g., magnetorquers), subclasses
        should override this method to account for environmental factors.
        
        Parameters
        ----------
        orbital_states : list of Orbital_State, optional
            Sample orbital states for environment-dependent estimation.
            If None, uses expected_field_uT for a typical estimate.
        n_samples : int, optional
            Number of random samples if no orbital_states provided. Default 100.
        expected_field_uT : float, optional
            Expected magnetic field magnitude in μT for default estimation.
            Only used by environment-dependent actuators. Default 30.0.
            
        Returns
        -------
        float
            Estimated maximum torque magnitude in N·m.
            
        Notes
        -----
        This method is used by the planner's auto-scaling feature to normalize
        actuator costs. Weak actuators (low torque capability) get lower costs
        to encourage the optimizer to use them at full capacity.
        
        The base implementation assumes torque = u * axis, so max torque = u_max.
        """
        return self.u_max

    def torque(self, u: float, x: np.ndarray, os: Orbital_State, dmode: ErrorMode = None) -> float:
        r"""
        Compute the body-frame torque produced by the actuator.

        .. math::

            \boldsymbol{\tau}
            =
            \boldsymbol{\tau}(u, \mathbf{x}, \mathrm{os})

        This base implementation returns zero torque.

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state
                  :math:`[\boldsymbol{\omega};\mathbf{q}]`.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state providing environmental context.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param dmode: Optional disturbance configuration.
        :type dmode: :class:`~ADCS.satellite_hardware.errors.error_mode.ErrorMode` or None

        :return: Body-frame torque vector.
        :rtype: numpy.ndarray of shape ``(3,)``
        """
        return np.ndarray([0, 0, 0])
    
    def storage_torque(self, u: float, j2000: float, dmode: ErrorMode = None)-> float:
        r"""
        Compute the torque contribution associated with momentum storage states.

        This is relevant for actuators such as reaction wheels that exchange
        angular momentum with an internal state.

        Base class has no storage degrees of freedom.

        :param u: Actuator command input.
        :type u: float

        :param j2000: Absolute time in seconds since J2000 epoch.
        :type j2000: float

        :param dmode: Optional disturbance configuration.
        :type dmode: :class:`~ADCS.satellite_hardware.errors.error_mode.ErrorMode` or None

        :return: Storage torque vector.
        :rtype: numpy.ndarray of shape ``(0,)``
        """
        return np.zeros((0,))
    
    def dtorq__du(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of torque with respect to input.

        .. math::

            \frac{\partial \boldsymbol{\tau}}{\partial u}
            \in \mathbb{R}^{1\times 3}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Row-Jacobian of torque w.r.t. input.
        :rtype: numpy.ndarray of shape ``(1, 3)``
        """
        return np.zeros((1, 3))
    
    def dtorq__dbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of torque with respect to actuator bias.

        If a bias model exists, this mirrors
        :meth:`~ADCS.satellite_hardware.actuators.actuator.Actuator.dtorq__du`.

        .. math::

            \frac{\partial \boldsymbol{\tau}}{\partial \mathbf{b}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Row-Jacobian of torque w.r.t. bias.
        :rtype: numpy.ndarray of shape ``(1, 3)`` or ``(0, 3)``
        """
        if self.bias:
            return self.dtorq__du(u=u, x=x, os=os)
        else:
            return np.zeros((0, 3))
        
    def dtorq__dbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of torque with respect to the base spacecraft state.

        .. math::

            \frac{\partial \boldsymbol{\tau}}{\partial \mathbf{x}}
            =
            \begin{bmatrix}
                \frac{\partial}{\partial \boldsymbol{\omega}} \\
                \frac{\partial}{\partial \mathbf{q}}
            \end{bmatrix}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Row-Jacobian of torque w.r.t. base state.
        :rtype: numpy.ndarray of shape ``(7, 3)``
        """
        return np.zeros((7, 3))
    
    def dtorq__dh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of torque with respect to momentum storage state.

        Base class has no storage state.

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Row-Jacobian of torque w.r.t. storage state.
        :rtype: numpy.ndarray of shape ``(0, 3)``
        """
        return np.zeros((0,3))
    
    def ddtorq__dudu(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative of torque with respect to input.

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}}{\partial u^2}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Input–input Hessian of torque.
        :rtype: numpy.ndarray of shape ``(1, 1, 3)``
        """
        return np.zeros((1, 1, 3))
    
    def ddtorq__dudbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of torque with respect to input and bias.

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}}{\partial u\,\partial \mathbf{b}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Input–bias Hessian of torque.
        :rtype: numpy.ndarray of shape ``(1, 1, 3)`` or ``(1, 0, 3)``
        """
        if self.bias:
            return self.ddtorq__dudu(u=u, x=x, os=os)
        else:
            return np.zeros((1, 0, 3))
        
    def ddtorq__dudbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of torque with respect to input and base state.

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}}{\partial u\,\partial \mathbf{x}}
            =
            \begin{bmatrix}
                \frac{\partial^2 \boldsymbol{\tau}}{\partial u\,\partial \boldsymbol{\omega}} \\
                \frac{\partial^2 \boldsymbol{\tau}}{\partial u\,\partial \mathbf{q}}
            \end{bmatrix}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Input–state Hessian of torque.
        :rtype: numpy.ndarray of shape ``(1, 7, 3)``
        """

        return np.zeros((1, 7, 3))
    
    def ddtorq__dudh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of torque with respect to input and storage state.

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}}{\partial u\,\partial \mathbf{h}}

        Base class has no storage state.

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Input–storage Hessian of torque.
        :rtype: numpy.ndarray of shape ``(1, 0, 3)``
        """
        return np.zeros((1, 0, 3))
    
    def ddtorq__dbiasdbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative of torque with respect to actuator bias.

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}}{\partial \mathbf{b}^2}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Bias–bias Hessian of torque.
        :rtype: numpy.ndarray of shape ``(1, 1, 3)`` or ``(0, 0, 3)``
        """

        if self.bias:
            return self.ddtorq__dudu(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0, 3))
        
    def ddtorq__dbiasdbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of torque with respect to bias and base state.

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}}{\partial \mathbf{b}\,\partial \mathbf{x}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Bias–state Hessian of torque.
        :rtype: numpy.ndarray of shape ``(1, 7, 3)`` or ``(0, 7, 3)``
        """
        if self.bias:
            return self.ddtorq__dudbasestate(u=u, x=x, os=os)
        else:
            return np.zeros((0, 7, 3))
        
    def ddtorq__dbiasdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of torque with respect to bias and storage state.

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}}{\partial \mathbf{b}\,\partial \mathbf{h}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Bias–storage Hessian of torque.
        :rtype: numpy.ndarray of shape ``(1, 0, 3)`` or ``(0, 0, 3)``
        """
        if self.bias:
            return self.ddtorq__dudh(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0, 3))
        
    def ddtorq__dbasestatedbasestate(self, u: float, x: np.ndarray, os: Orbital_State):
        r"""
        Second derivative of torque with respect to the base spacecraft state.

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}}{\partial \mathbf{x}^2}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: State–state Hessian of torque.
        :rtype: numpy.ndarray of shape ``(7, 7, 3)``
        """
        return np.zeros((7,7,3))
        
    def ddtorq__dbasestatedh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of torque with respect to base state and storage state.

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}}{\partial \mathbf{x}\,\partial \mathbf{h}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: State–storage Hessian of torque.
        :rtype: numpy.ndarray of shape ``(7, 0, 3)``
        """
        return np.zeros((7, 0, 3))
    
    def ddtorq__dhdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative of torque with respect to storage state.

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}}{\partial \mathbf{h}^2}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Storage–storage Hessian of torque.
        :rtype: numpy.ndarray of shape ``(0, 0, 3)``
        """
        return np.zeros((0, 0, 3))
    
    def dstor_torq__du(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of storage torque with respect to input.

        .. math::

            \frac{\partial \mathbf{t}_s}{\partial u}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Row-Jacobian of storage torque w.r.t. input.
        :rtype: numpy.ndarray of shape ``(1, 0)``
        """
        return np.zeros((1, 0))
    
    def dstor_torq__dbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of storage torque with respect to actuator bias.

        .. math::

            \frac{\partial \mathbf{t}_s}{\partial \mathbf{b}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Row-Jacobian of storage torque w.r.t. bias.
        :rtype: numpy.ndarray of shape ``(1, 0)`` or ``(0, 0)``
        """

        if self.bias:
            return self.dstor_torq__du(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0))
        
    def dstor_torq__dbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of storage torque with respect to the base spacecraft state.

        .. math::

            \frac{\partial \mathbf{t}_s}{\partial \mathbf{x}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Row-Jacobian of storage torque w.r.t. base state.
        :rtype: numpy.ndarray of shape ``(7, 0)``
        """
        return np.zeros((7, 0))
        
    def dstor_torq__dh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of storage torque with respect to storage state.

        .. math::

            \frac{\partial \mathbf{t}_s}{\partial \mathbf{h}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Row-Jacobian of storage torque w.r.t. storage state.
        :rtype: numpy.ndarray of shape ``(0, 0)``
        """
        return np.zeros((0, 0))
    
    def ddstor_torq__dudu(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative of storage torque with respect to input.

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Input–input Hessian of storage torque.
        :rtype: numpy.ndarray of shape ``(1, 1, 0)``
        """
        return np.zeros((1, 1, 0))
    
    def ddstor_torq__dudbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of storage torque with respect to input and bias.

        .. math::

            \frac{\partial^2 \mathbf{t}_s}{\partial u\,\partial \mathbf{b}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Input–bias Hessian of storage torque.
        :rtype: numpy.ndarray of shape ``(1, 1, 0)`` or ``(1, 0, 0)``
        """
        if self.bias:
            return self.ddstor_torq__dudu(u=u, x=x, os=os)
        else:
            return np.zeros((1, 0, 0))

    def ddstor_torq__dudbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of storage torque with respect to input and base state.

        .. math::

            \frac{\partial^2 \mathbf{t}_s}{\partial u\,\partial \mathbf{x}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Input–state Hessian of storage torque.
        :rtype: numpy.ndarray of shape ``(1, 7, 0)``
        """
        return np.zeros((1, 7, 0))
    
    def ddstor_torq__dudh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of storage torque with respect to input and storage state.

        .. math::

            \frac{\partial^2 \mathbf{t}_s}{\partial u\,\partial \mathbf{h}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Input–storage Hessian of storage torque.
        :rtype: numpy.ndarray of shape ``(1, 0, 0)``
        """
        return np.zeros((1, 0, 0))
    
    def ddstor_torq__dbiasdbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative of storage torque with respect to actuator bias.

        .. math::

            \frac{\partial^2 \mathbf{t}_s}{\partial \mathbf{b}^2}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Bias–bias Hessian of storage torque.
        :rtype: numpy.ndarray of shape ``(1, 1, 0)`` or ``(0, 0, 0)``
        """
        if self.bias:
            return self.ddstor_torq__dudu(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0, 0))
        
    def ddstor_torq__dbiasdbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of storage torque with respect to bias and base state.

        .. math::

            \frac{\partial^2 \mathbf{t}_s}{\partial \mathbf{b}\,\partial \mathbf{x}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Bias–state Hessian of storage torque.
        :rtype: numpy.ndarray of shape ``(1, 7, 0)`` or ``(0, 7, 0)``
        """
        if self.bias:
            return self.ddstor_torq__dudbasestate(u=u, x=x, os=os)
        else:
            return np.zeros((0, 7, 0))
        
    def ddstor_torq__dbiasdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of storage torque with respect to bias and storage state.

        .. math::

            \frac{\partial^2 \mathbf{t}_s}{\partial \mathbf{b}\,\partial \mathbf{h}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Bias–storage Hessian of storage torque.
        :rtype: numpy.ndarray of shape ``(1, 0, 0)`` or ``(0, 0, 0)``
        """
        if self.bias:
            return self.ddstor_torq__dudh(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0, 0))
        
    def ddstor_torq__dbasestatedbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative of storage torque with respect to the base spacecraft state.

        .. math::

            \frac{\partial^2 \mathbf{t}_s}{\partial \mathbf{x}^2}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: State–state Hessian of storage torque.
        :rtype: numpy.ndarray of shape ``(7, 7, 0)``
        """
        return np.zeros((7, 7, 0))
    
    def ddstor_torq__dbasestatedh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative of storage torque with respect to base state and storage state.

        .. math::

            \frac{\partial^2 \mathbf{t}_s}{\partial \mathbf{x}\,\partial \mathbf{h}}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: State–storage Hessian of storage torque.
        :rtype: numpy.ndarray of shape ``(7, 0, 0)``
        """
        return np.zeros((7, 0, 0))
    
    def ddstor_torq__dhdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative of storage torque with respect to storage state.

        .. math::

            \frac{\partial^2 \mathbf{t}_s}{\partial \mathbf{h}^2}

        :param u: Actuator command input.
        :type u: float

        :param x: Base spacecraft state.
        :type x: numpy.ndarray of shape ``(7,)``

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Storage–storage Hessian of storage torque.
        :rtype: numpy.ndarray of shape ``(0, 0, 0)``
        """
        return np.zeros((0, 0, 0))
    


    
