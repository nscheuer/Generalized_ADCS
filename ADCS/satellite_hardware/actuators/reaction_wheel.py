__all__ = ["RW"]

import numpy as np
import warnings
from ADCS.satellite_hardware.actuators.actuator import Actuator
from ADCS.satellite_hardware.errors.bias import Bias
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.orbits.orbital_state import Orbital_State

class RW(Actuator):
    r"""
    **Reaction Wheel (RW) Actuator Model**

    This class models a single-axis reaction wheel used for spacecraft attitude control.
    A reaction wheel generates a body torque by accelerating/decelerating its rotor, exchanging
    angular momentum with the spacecraft through conservation of angular momentum.

    **Torque model**

    Let :math:`\mathbf{a}\in\mathbb{R}^3` be the wheel spin axis expressed in the spacecraft body frame,
    with :math:`\|\mathbf{a}\|=1`. Let :math:`u` be the commanded motor torque (scalar) about the wheel axis.

    The body torque applied by the wheel is

    .. math::

        \boldsymbol{\tau}_{\mathrm{RW}} = \mathbf{a}\,u.

    If actuator bias and noise are enabled, the effective commanded torque becomes

    .. math::

        u_{\mathrm{eff}} = u + b(t) + n(t),

    yielding

    .. math::

        \boldsymbol{\tau}_{\mathrm{RW}} = \mathbf{a}\,\bigl(u + b(t) + n(t)\bigr).

    **Wheel momentum dynamics**

    Let :math:`\mathbf{h}\in\mathbb{R}^3` denote the stored wheel angular momentum vector in the body frame.
    For a single-axis wheel, :math:`\mathbf{h}` is collinear with :math:`\mathbf{a}`. By conservation of angular
    momentum, the internal wheel torque is equal and opposite to the body torque:

    .. math::

        \boldsymbol{\tau}_{\mathrm{wheel}} = -\boldsymbol{\tau}_{\mathrm{RW}}.

    Consequently, the wheel momentum changes according to

    .. math::

        \dot{\mathbf{h}} = -\boldsymbol{\tau}_{\mathrm{RW}}.

    The wheel momentum is limited by a saturation bound :math:`\|\mathbf{h}\|\le h_{\max}` (componentwise
    in this implementation).

    **Measurement model**

    Wheel momentum measurements are modeled as

    .. math::

        \tilde{\mathbf{h}} = \mathbf{h} + \nu_h,

    where :math:`\nu_h` is provided by :class:`~ADCS.satellite_hardware.errors.noise.Noise`.

    **Related types**

    - Inherits from :class:`~ADCS.satellite_hardware.actuators.actuator.Actuator`.
    - Bias model: :class:`~ADCS.satellite_hardware.errors.bias.Bias`.
    - Noise model: :class:`~ADCS.satellite_hardware.errors.noise.Noise`.
    - Disturbance toggles: :class:`~ADCS.satellite_hardware.errors.error_mode.ErrorMode`.
    - Orbital state (for time-tagging): :class:`~ADCS.orbits.orbital_state.Orbital_State`.
    """

    def __init__(
        self,
        axis: np.ndarray,
        max_torque: float,
        J: float,
        h: np.ndarray,
        h_max: np.ndarray,
        bias: Bias = None,
        noise: Noise = None,
        h_meas_noise: Noise = None,
        estimate_bias: bool = False,
    ) -> None:
        r"""
        Construct a reaction wheel actuator model.

        The wheel is defined by its body-frame spin axis :math:`\mathbf{a}`, motor torque limit
        :math:`u_{\max}`, wheel inertia :math:`J`, current stored momentum :math:`\mathbf{h}`, and a
        momentum saturation bound :math:`\mathbf{h}_{\max}`.

        If provided, the bias and noise models are applied to the scalar commanded torque before it is
        mapped onto the body-frame torque vector along :math:`\mathbf{a}`.

        The optional momentum measurement noise :math:`\nu_h` is used in :meth:`~RW.measure_momentum` to
        produce a noisy measurement :math:`\tilde{\mathbf{h}}`.

        .. math::

            \boldsymbol{\tau}_{\mathrm{RW}} = \mathbf{a}\,\bigl(u + b(t) + n(t)\bigr), \qquad
            \tilde{\mathbf{h}} = \mathbf{h} + \nu_h.

        :param axis: Unit spin axis of the reaction wheel in the spacecraft body frame, shape ``(3,)``.
        :type axis: numpy.ndarray

        :param max_torque: Maximum motor torque magnitude :math:`u_{\max}` the wheel can produce [N·m].
        :type max_torque: float

        :param J: Wheel rotor moment of inertia [kg·m²].
        :type J: float

        :param h: Current wheel angular momentum vector :math:`\mathbf{h}` [N·m·s], shape ``(3,)``.
        :type h: numpy.ndarray

        :param h_max: Maximum allowable angular momentum :math:`\mathbf{h}_{\max}` [N·m·s], shape ``(3,)``.
        :type h_max: numpy.ndarray

        :param bias: Optional actuator bias model providing :math:`b(t)` applied to the scalar commanded torque.
        :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` | None

        :param noise: Optional actuator torque noise model providing :math:`n(t)` applied to the scalar commanded torque.
        :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` | None

        :param h_meas_noise: Optional measurement noise model for momentum measurement, providing :math:`\nu_h`.
        :type h_meas_noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` | None

        :param estimate_bias: If ``True``, include this actuator’s bias in the estimator state.
        :type estimate_bias: bool

        :return: ``None``.
        :rtype: NoneType
        """
        self.J = J
        self.h = h
        self.h_max = h_max

        if h_meas_noise:
            self.h_meas_noise: Noise = h_meas_noise
        else:
            self.h_meas_noise = Noise()
        super().__init__(axis=axis, u_max=max_torque, bias=bias, noise=noise, estimate_bias=estimate_bias)

    def torque(self, u: float, x: np.ndarray, os: Orbital_State, dmode: ErrorMode = None) -> float:
        r"""
        Compute the reaction-wheel torque vector applied to the spacecraft body.

        The commanded scalar motor torque :math:`u` is mapped onto the wheel axis :math:`\mathbf{a}` to
        produce a 3D body torque. If enabled via :class:`~ADCS.satellite_hardware.errors.error_mode.ErrorMode`,
        additive bias :math:`b(t)` and noise :math:`n(t)` are applied to the scalar torque before mapping.

        .. math::

            u_{\mathrm{eff}} = u + b(t) + n(t),

        .. math::

            \boldsymbol{\tau}_{\mathrm{RW}} = \mathbf{a}\,u_{\mathrm{eff}}.

        Bias evolution (if present) may depend on the orbital time tag :math:`t`, provided here through
        :attr:`~ADCS.orbits.orbital_state.Orbital_State.J2000`.

        If :math:`|u| > u_{\max}` a warning is emitted; the returned torque is not clipped.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft state vector (passed through for interface compatibility).
        :type x: numpy.ndarray

        :param os: Orbital state providing time tag (e.g., :attr:`~ADCS.orbits.orbital_state.Orbital_State.J2000`) for bias evolution.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param dmode: Disturbance mode toggles controlling inclusion and update of bias and noise.
        :type dmode: :class:`~ADCS.satellite_hardware.errors.error_mode.ErrorMode` | None

        :return: Body torque vector :math:`\boldsymbol{\tau}_{\mathrm{RW}}` [N·m], shape ``(3,)``.
        :rtype: numpy.ndarray
        """
        if abs(u) > self.u_max:
            warnings.warn("requested torque exceeds actuation limit")

        if dmode is None:
            dmode = ErrorMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)

        torque = u

        # Evolve states first so this call uses values at the current step.
        if self.bias and dmode.update_bias:
            self.bias._update_bias(j2000=os.J2000)
        if self.noise and dmode.update_noise:
            self.noise._update_noise()

        if self.bias and dmode.add_bias:
            torque += self.bias.get_bias(j2000=os.J2000)

        if self.noise and dmode.add_noise:
            torque += self.noise.get_noise()

        return self.axis*torque

    def storage_torque(self, u: float, x: np.ndarray, os: Orbital_State, dmode: ErrorMode = None) -> float:
        r"""
        Compute the internal torque acting on the wheel (equal and opposite to the body torque).

        The wheel experiences the opposite torque to that applied to the spacecraft body. Using the same
        effective scalar command :math:`u_{\mathrm{eff}} = u + b(t) + n(t)` (when enabled), the internal
        wheel torque is

        .. math::

            \boldsymbol{\tau}_{\mathrm{wheel}} = -\boldsymbol{\tau}_{\mathrm{RW}}
            = -\mathbf{a}\,u_{\mathrm{eff}}.

        This quantity is the torque that drives the wheel momentum dynamics:

        .. math::

            \dot{\mathbf{h}} = \boldsymbol{\tau}_{\mathrm{wheel}}.

        If :math:`|u| > u_{\max}` a warning is emitted; the returned value is not clipped.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft state vector (passed through for interface compatibility).
        :type x: numpy.ndarray

        :param os: Orbital state providing time tag (e.g., :attr:`~ADCS.orbits.orbital_state.Orbital_State.J2000`) for bias evolution.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param dmode: Disturbance mode toggles controlling inclusion and update of bias and noise.
        :type dmode: :class:`~ADCS.satellite_hardware.errors.error_mode.ErrorMode` | None

        :return: Internal wheel torque vector :math:`\boldsymbol{\tau}_{\mathrm{wheel}}` [N·m], shape ``(3,)``.
        :rtype: numpy.ndarray
        """

        if abs(u) > self.u_max:
            warnings.warn("RW Requested Torque exceeds actuation limit")

        if dmode is None:
            dmode = ErrorMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)
        
        command = u

        # Evolve stochastic states first so this call uses values at the current step.
        if self.bias and dmode.update_bias:
            self.bias._update_bias(j2000=os.J2000)
        if self.noise and dmode.update_noise:
            self.noise._update_noise()

        if self.bias and dmode.add_bias:
            command += self.bias.get_bias(j2000=os.J2000)

        if self.noise and dmode.add_noise:
            command += self.noise.get_noise()

        return -command
    

    def measure_momentum(self):
        r"""
        Measure the wheel angular momentum with additive measurement noise.

        The measurement model is

        .. math::

            \tilde{\mathbf{h}} = \mathbf{h} + \nu_h,

        where :math:`\nu_h` is sampled from :attr:`~RW.h_meas_noise`, an instance of
        :class:`~ADCS.satellite_hardware.errors.noise.Noise`.

        :return: Noisy wheel momentum measurement :math:`\tilde{\mathbf{h}}` [N·m·s], shape ``(3,)``.
        :rtype: numpy.ndarray
        """
        return self.h + self.h_meas_noise.get_noise()
    
    def measure_momentum_noiseless(self):
        r"""
        Measure the wheel angular momentum without measurement noise.

        This returns the stored wheel momentum state directly:

        .. math::

            \tilde{\mathbf{h}} = \mathbf{h}.

        :return: Noiseless wheel momentum :math:`\mathbf{h}` [N·m·s], shape ``(3,)``.
        :rtype: numpy.ndarray
        """
        return self.h
        
    def update_momentum(self, h: float) -> None:
        r"""
        Update the stored wheel angular momentum.

        This method overwrites the internal momentum state :math:`\mathbf{h}`. If the provided value
        exceeds the saturation bound, a warning is emitted; the value is still assigned.

        .. math::

            \mathbf{h} \leftarrow \mathbf{h}_{\text{new}}.

        :param h: New wheel angular momentum [N·m·s].
        :type h: float

        :return: ``None``.
        :rtype: NoneType
        """
        if h > self.h_max:
            warnings.warn("RW Angular Momentum exceeds saturation limit")
        self.h = h

    def dtorq__du(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian :math:`\partial\boldsymbol{\tau}/\partial u`.

        The body torque is

        .. math::

            \boldsymbol{\tau}(u) = \mathbf{a}\,u_{\mathrm{eff}}, \qquad
            u_{\mathrm{eff}} = u + b(t) + n(t),

        and the mapping from scalar command :math:`u` to torque is linear. Therefore,

        .. math::

            \frac{\partial\boldsymbol{\tau}}{\partial u} = \mathbf{a}.

        This implementation returns a row-shaped Jacobian consistent with a scalar input and a 3D torque output.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Jacobian :math:`\partial\boldsymbol{\tau}/\partial u`, shape ``(1,3)``.
        :rtype: numpy.ndarray
        """
        return self.axis.reshape((1,3))

    def dtorq__dh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian :math:`\partial\boldsymbol{\tau}/\partial \mathbf{h}`.

        In this model, the applied body torque :math:`\boldsymbol{\tau}` depends on the commanded torque
        and (optionally) actuator bias/noise, but not on the stored wheel momentum state :math:`\mathbf{h}`.
        Hence,

        .. math::

            \frac{\partial\boldsymbol{\tau}}{\partial \mathbf{h}} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero Jacobian :math:`\partial\boldsymbol{\tau}/\partial \mathbf{h}`, shape ``(1,3)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((1,3))
    
    def ddtorq__dudh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}/\partial u\,\partial \mathbf{h}`.

        The torque does not depend on :math:`\mathbf{h}`, so the mixed second derivative is identically zero:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial u\,\partial \mathbf{h}} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor, shape ``(1,1,3)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((1, 1, 3))
    
    def ddtorq__dbiasdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}/\partial b\,\partial \mathbf{h}`.

        The torque does not depend on :math:`\mathbf{h}`, so the mixed derivative with respect to bias and
        momentum is zero:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial b\,\partial \mathbf{h}} = \mathbf{0}.

        If a bias model exists, the returned tensor is the appropriately shaped zero tensor. If no bias
        model exists, an empty tensor consistent with the absence of bias states is returned.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor if a bias model exists; otherwise an empty tensor consistent with no bias state.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return self.ddtorq__dudh(u=u, x=x, os=os)
        else:
            return np.zeros((0, 1, 3))
        
    def ddtorq__dbasestatedh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}/\partial \mathbf{x}_{\mathrm{base}}\,\partial \mathbf{h}`.

        The torque model :math:`\boldsymbol{\tau}=\mathbf{a}\,u_{\mathrm{eff}}` does not depend on the spacecraft
        base state :math:`\mathbf{x}_{\mathrm{base}}` nor on :math:`\mathbf{h}`. Therefore the mixed second derivative
        is identically zero:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial \mathbf{x}_{\mathrm{base}}\,\partial \mathbf{h}} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor, shape ``(7,1,3)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((7, 1, 3))
    
    def ddtorq__dhdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}/\partial \mathbf{h}^2`.

        The torque does not depend on the stored wheel momentum :math:`\mathbf{h}`, so:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial \mathbf{h}^2} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor, shape ``(1,1,3)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((1, 1, 3))
    
    def dstor_torq__du(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian :math:`\partial\boldsymbol{\tau}_{\mathrm{wheel}}/\partial u`.

        The internal wheel torque is

        .. math::

            \boldsymbol{\tau}_{\mathrm{wheel}} = -\mathbf{a}\,u_{\mathrm{eff}}, \qquad
            u_{\mathrm{eff}} = u + b(t) + n(t).

        Differentiating with respect to :math:`u` yields

        .. math::

            \frac{\partial\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial u} = -\mathbf{a}.

        In this implementation, the storage torque is returned as a scalar command-like quantity, so the Jacobian
        with respect to the scalar input is a ``(1,1)`` matrix equal to :math:`-1`.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Jacobian :math:`\partial\boldsymbol{\tau}_{\mathrm{wheel}}/\partial u`, shape ``(1,1)``.
        :rtype: numpy.ndarray
        """
        return -np.ones((1,1))
    
    def dstor_torq__dbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian :math:`\partial\boldsymbol{\tau}_{\mathrm{wheel}}/\partial b`.

        The bias :math:`b` enters the effective command linearly through :math:`u_{\mathrm{eff}} = u + b + n`,
        and the internal wheel torque is :math:`\boldsymbol{\tau}_{\mathrm{wheel}} = -\mathbf{a}\,u_{\mathrm{eff}}`.
        Therefore,

        .. math::

            \frac{\partial\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial b} = -\mathbf{a}.

        In this implementation, the storage torque is represented in scalar form, so the derivative is ``-1``
        when a bias state exists; otherwise an empty tensor consistent with no bias state is returned.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: ``(1,1)`` matrix equal to :math:`-1` if a bias model exists; otherwise an empty array with shape ``(0,1)``.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return self.dstor_torq__du(u=u, x=x, os=os)
        else:
            return np.zeros((0, 1))
        
    def dstor_torq__dbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian :math:`\partial\boldsymbol{\tau}_{\mathrm{wheel}}/\partial \mathbf{x}_{\mathrm{base}}`.

        The internal wheel torque depends only on the commanded torque (and optional bias/noise), not on the spacecraft
        base state :math:`\mathbf{x}_{\mathrm{base}}`. Hence,

        .. math::

            \frac{\partial\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial \mathbf{x}_{\mathrm{base}}} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero Jacobian, shape ``(7,1)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((7, 1))
    
    def dstor_torq__dh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Jacobian :math:`\partial\boldsymbol{\tau}_{\mathrm{wheel}}/\partial \mathbf{h}`.

        The internal wheel torque does not depend on the stored momentum :math:`\mathbf{h}` in this actuator model,
        so:

        .. math::

            \frac{\partial\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial \mathbf{h}} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero Jacobian, shape ``(1,1)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((1, 1))
    
    def ddstor_torq__dudu(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}/\partial u^2`.

        The internal wheel torque depends linearly on :math:`u`:

        .. math::

            \boldsymbol{\tau}_{\mathrm{wheel}}(u) = -\mathbf{a}\,(u+b+n),

        so its second derivative with respect to :math:`u` is zero:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial u^2} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor, shape ``(1,1,1)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((1, 1, 1))
    
    def ddstor_torq__dudbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}/\partial u\,\partial b`.

        The bias :math:`b` enters linearly through :math:`(u+b)`, so the mixed second derivative is zero:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial u\,\partial b} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor if a bias model exists; otherwise an empty tensor consistent with no bias state.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return self.ddstor_torq__dudu(u=u, x=x, os=os)
        else:
            return np.zeros((1, 0, 1))
        
    def ddstor_torq__dudbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}/\partial u\,\partial \mathbf{x}_{\mathrm{base}}`.

        The internal wheel torque does not depend on the base state :math:`\mathbf{x}_{\mathrm{base}}`, and is linear in
        :math:`u`. Therefore the mixed second derivative is zero:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial u\,\partial \mathbf{x}_{\mathrm{base}}} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor, shape ``(1,7,1)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((1, 7, 1))
    
    def ddstor_torq__dudh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}/\partial u\,\partial \mathbf{h}`.

        The internal wheel torque does not depend on :math:`\mathbf{h}`, hence:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial u\,\partial \mathbf{h}} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor, shape ``(1,1,1)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((1, 1, 1))
    
    def ddstor_torq__dbiasdbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}/\partial b^2`.

        The bias :math:`b` enters the internal torque linearly through :math:`(u+b)`, so:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial b^2} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor if a bias model exists; otherwise an empty tensor consistent with no bias state.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return self.ddstor_torq__dudu(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0, 1))
        
    def ddstor_torq__dbiasdbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}/\partial b\,\partial \mathbf{x}_{\mathrm{base}}`.

        The internal wheel torque is independent of the base state and linear in :math:`b`, hence:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial b\,\partial \mathbf{x}_{\mathrm{base}}} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor if a bias model exists; otherwise an empty tensor consistent with no bias state.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return self.ddstor_torq__dudbasestate(u=u, x=x, os=os)
        else:
            return np.zeros((0, 7, 1))
        
    def ddstor_torq__dbiasdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}/\partial b\,\partial \mathbf{h}`.

        The internal wheel torque does not depend on :math:`\mathbf{h}`, so:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial b\,\partial \mathbf{h}} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor if a bias model exists; otherwise an empty tensor consistent with no bias state.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return self.ddstor_torq__dudh(u=u, x=x, os=os)
        else:
            return np.zeros((0, 1, 1))

    def ddstor_torq__dbasestatedbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}/\partial \mathbf{x}_{\mathrm{base}}^2`.

        The internal wheel torque is independent of the base state :math:`\mathbf{x}_{\mathrm{base}}`, so:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial \mathbf{x}_{\mathrm{base}}^2} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor, shape ``(7,7,1)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((7, 7, 1))
    
    def ddstor_torq__dbasestatedh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}/\partial \mathbf{x}_{\mathrm{base}}\,\partial \mathbf{h}`.

        The internal wheel torque does not depend on :math:`\mathbf{x}_{\mathrm{base}}` nor :math:`\mathbf{h}`, hence:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial \mathbf{x}_{\mathrm{base}}\,\partial \mathbf{h}} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor, shape ``(7,1,1)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((7, 1, 1))
    
    def ddstor_torq__dhdh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}/\partial \mathbf{h}^2`.

        The internal wheel torque is independent of the stored momentum :math:`\mathbf{h}`, so:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}_{\mathrm{wheel}}}{\partial \mathbf{h}^2} = \mathbf{0}.

        :param u: Commanded motor torque about the wheel axis [N·m].
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor, shape ``(1,1,1)``.
        :rtype: numpy.ndarray
        """
        return np.zeros((1, 1, 1))