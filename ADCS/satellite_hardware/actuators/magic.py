__all__ = ["Magic_Actuator"]

import numpy as np
import warnings
from typing import Dict, Union

from ADCS.satellite_hardware.actuators.actuator import Actuator
from ADCS.satellite_hardware.errors.bias import Bias
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.orbits.orbital_state import Orbital_State


class Magic_Actuator(Actuator):
    r"""
    **Magic (ideal direct body-torque) actuator model.**

    A "magic" actuator applies a body-frame torque directly along its body
    axis with no environmental or state coupling. The torque is

    .. math::

        \boldsymbol{\tau}_{\mathrm{magic}}
        = \mathbf{a}\,(u + b) + \mathbf{n},

    where :math:`\mathbf{a}` is the body-axis unit vector, :math:`u` the
    commanded torque magnitude, :math:`b` an optional scalar bias, and
    :math:`\mathbf{n}` an optional additive noise.

    Why "magic"
    -----------
    Unlike a magnetorquer (whose torque depends on the geomagnetic field and
    therefore on the spacecraft attitude) or a reaction wheel (which has its
    own momentum state and exchanges angular momentum via Newton's 3rd law),
    the magic actuator is dynamically trivial:

    * No state coupling: no attitude dependence, no momentum exchange.
    * No environmental dependence: torque does not depend on B-field, R, V, S.
    * Constant Jacobian: :math:`\partial\boldsymbol{\tau}/\partial u = \mathbf{a}`.

    This makes the magic actuator ideal for tests: it provides a "clean"
    body-torque commander that doesn't carry the MTQ ``m \times B`` rank
    deficiency or the RW back-reaction inertia / wheel-momentum state.

    Compatibility
    -------------
    The C++ OldPlanner (``trajectory_planner/``) has full magic-actuator
    support via ``pysat.Satellite.add_magic(axis, max_torq, cost)``. The
    SALTRO planner has a vestigial ``magic_control_weight`` cost setting
    but no actuator class -- that side of the parity is a separate fix.

    Symbols
    -------

    .. list-table::
       :header-rows: 1
       :widths: 30 70

       * - Symbol
         - Meaning
       * - :math:`\mathbf{a}`
         - Magic actuator unit axis in the body frame
       * - :math:`u`
         - Commanded torque magnitude [N·m]
       * - :math:`b`
         - Optional scalar bias
       * - :math:`\mathbf{n}`
         - Optional additive noise

    :param axis: Unit vector defining the magic actuator's body axis.
    :type axis: numpy.ndarray, shape ``(3,)``

    :param max_torque: Maximum allowable commanded torque magnitude.
    :type max_torque: float

    :param bias: Optional bias model.
    :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` or None

    :param noise: Optional noise model.
    :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` or None

    :param estimate_bias: If True, the bias enters the estimator state.
    :type estimate_bias: bool
    """

    def __init__(
        self,
        axis: np.ndarray,
        max_torque: float,
        bias: Bias = None,
        noise: Noise = None,
        estimate_bias: bool = False,
    ) -> None:
        super().__init__(axis=axis, u_max=max_torque, bias=bias, noise=noise,
                         estimate_bias=estimate_bias)

    def torque(
        self,
        u: float,
        x: np.ndarray,
        os: Union[Orbital_State, Dict[str, np.ndarray]],
        dmode: ErrorMode = None,
    ) -> np.ndarray:
        r"""Body-frame torque ``τ = a (u + b) + n``.

        Direct application of the commanded magnitude along the actuator's
        body axis. No attitude or environment dependence.
        """
        if abs(u) > self.u_max:
            warnings.warn("requested torque exceeds actuation limit")

        if dmode is None:
            dmode = ErrorMode(add_bias=True, add_noise=True,
                              update_bias=True, update_noise=True)

        u_eff = float(np.asarray(u).reshape(-1)[0])
        if self.bias and dmode.add_bias:
            u_eff += self.bias.get_bias(j2000=os.J2000)
        if dmode.update_bias:
            self.bias._update_bias(j2000=os.J2000)

        torque = self.axis * u_eff

        if self.noise and dmode.add_noise:
            torque = torque + self.noise.get_noise()
        if dmode.update_noise:
            self.noise._update_noise()

        return torque

    def storage_torque(
        self,
        u: float,
        x: np.ndarray,
        os: Orbital_State,
        dmode: ErrorMode = None,
    ) -> np.ndarray:
        """Magic actuators have no momentum-storage state.

        Returns the empty vector consistent with the base ``Actuator``
        convention.
        """
        return np.zeros((0,))

    # ------------------------------------------------------------------
    # First-order derivatives
    # ------------------------------------------------------------------

    def dtorq__du(
        self, u: float, x: np.ndarray, os: Orbital_State
    ) -> np.ndarray:
        r"""``∂τ/∂u = a`` (a constant 1x3 row).

        The torque is affine in ``u`` so the Jacobian is the actuator
        axis itself, independent of state or environment.
        """
        return self.axis.reshape((1, 3))

    def dtorq__dbias(
        self, u: float, x: np.ndarray, os: Orbital_State
    ) -> np.ndarray:
        r"""``∂τ/∂b = a`` if a bias model is attached.

        The bias enters identically to ``u`` (both as ``(u + b)``), so
        :math:`\partial\boldsymbol{\tau}/\partial b = \partial\boldsymbol{\tau}/\partial u = \mathbf{a}`.
        Without a bias model, an empty matrix is returned.
        """
        if self.bias:
            return self.axis.reshape((1, 3))
        return np.zeros((0, 3))

    # ------------------------------------------------------------------
    # Convenience: jacobians() / hessians() bundling (matches MTQ/RW API)
    # ------------------------------------------------------------------

    def jacobians(self, u, x, os):
        """Return ``(dT_du, dT_dx)`` matching MTQ/RW API.

        ``dT_du`` is ``(1, 3) = axis`` and ``dT_dx`` is ``(7, 3) =`` zeros
        because the torque has no base-state dependence.
        """
        dtorq__du = self.axis.reshape((1, 3))
        dtorq__dbasestate = np.zeros((7, 3))
        return dtorq__du, dtorq__dbasestate

    def hessians(self, u, x, os):
        """Return ``(ddT_du_dx, ddT_dx2)`` matching MTQ/RW API.

        Both are identically zero -- the magic torque is affine in ``u``
        and independent of ``x``, so all second derivatives vanish.
        """
        ddtorq__dudbasestate = np.zeros((1, 7, 3))
        ddtorq__dbasestatedbasestate = np.zeros((7, 7, 3))
        return ddtorq__dudbasestate, ddtorq__dbasestatedbasestate
