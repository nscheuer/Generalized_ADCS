__all__ = ["General_Disturbance"]

import numpy as np

from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.orbits.orbital_state import Orbital_State


class General_Disturbance(Disturbance):
    r"""
    **General (lumped) estimable disturbance torque.**

    Models an unmodeled, constant body-frame disturbance torque as a directly
    estimated 3-vector parameter :math:`\boldsymbol{\theta}`:

    .. math::

        \mathbf{T}_{\mathrm{gen}} = \boldsymbol{\theta},

    independent of attitude and orbital state. Because the torque *is* the
    estimated parameter, the parameter Jacobian is the identity and all
    attitude derivatives vanish:

    .. math::

        \frac{\partial \mathbf{T}}{\partial \boldsymbol{\theta}} = \mathbf{I}_3,
        \qquad
        \frac{\partial \mathbf{T}}{\partial \mathbf{q}} = \mathbf{0}_{4\times3}.

    With ``estimate_dist=True`` and ``estimated_vector_length=3`` this lets an
    augmented attitude estimator (UAKF/SRUAKF) carry the constant body torque
    as part of its state — e.g. to observe a cold-gas leak or other lumped
    unmodeled torque and hand the estimate to a planner. The estimated value is
    written back through the inherited ``main_param`` parameter interface.

    :param torque_init: Initial estimate of the disturbance torque [N·m], shape ``(3,)``.
        Defaults to zero.
    :type torque_init: :class:`numpy.ndarray` | None
    :param noise: Optional process-noise model for the random-walk ``update``.
    :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` | None
    :param estimate_dist: Include this disturbance's parameter in the estimator state.
    :type estimate_dist: bool
    """

    def __init__(self, torque_init: np.ndarray = None, noise: Noise = None,
                 estimate_dist: bool = True):
        super().__init__(estimate_dist=estimate_dist, estimated_vector_length=3)
        self._main_param = (np.zeros(3) if torque_init is None
                            else np.asarray(torque_init, dtype=float).reshape(3))
        self.noise = noise if noise is not None else Noise()
        self.current_torque = self._main_param.copy()

    # -- estimated-parameter interface (overrides the base property) ----------
    @property
    def main_param(self) -> np.ndarray:
        return self._main_param

    @main_param.setter
    def main_param(self, value) -> None:
        self._main_param = np.asarray(value, dtype=float).reshape(3)

    def update(self) -> None:
        r"""Random-walk the disturbance by one process-noise draw."""
        self.current_torque = self._main_param + self.noise.get_noise()

    # -- torque + derivatives (flexible signatures, matching Prop_Disturbance)--
    def torque(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""Disturbance torque = the estimated parameter, shape ``(3,)``."""
        return self._main_param

    def torque_valjac(self, *args, **kwargs) -> np.ndarray:
        r""":math:`\partial \mathbf{T}/\partial\boldsymbol{\theta} = \mathbf{I}_3`, shape ``(3, 3)``."""
        return np.eye(3)

    def torque_qjac(self, *args, **kwargs) -> np.ndarray:
        r""":math:`\partial \mathbf{T}/\partial\mathbf{q} = \mathbf{0}`, shape ``(4, 3)``."""
        return np.zeros((4, 3))

    def torque_qqhess(self, *args, **kwargs) -> np.ndarray:
        r""":math:`\partial^2 \mathbf{T}/\partial\mathbf{q}^2 = \mathbf{0}`, shape ``(4, 4, 3)``."""
        return np.zeros((4, 4, 3))

    def torque_valvalhess(self, *args, **kwargs) -> np.ndarray:
        r""":math:`\partial^2 \mathbf{T}/\partial\boldsymbol{\theta}^2 = \mathbf{0}`, shape ``(3, 3, 3)``."""
        return np.zeros((3, 3, 3))

    def torque_qvalhess(self, *args, **kwargs) -> np.ndarray:
        r""":math:`\partial^2 \mathbf{T}/\partial\mathbf{q}\,\partial\boldsymbol{\theta} = \mathbf{0}`, shape ``(4, 3, 3)``."""
        return np.zeros((4, 3, 3))
