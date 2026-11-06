__all__ = ["Thrust_Disturbance"]

import numpy as np

from typing import TYPE_CHECKING, Optional
from ADCS.state import State
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.orbits.orbital_state import Orbital_State

if TYPE_CHECKING:
    from ADCS.satellite_hardware.satellite.satellite import Satellite


class Thrust_Disturbance(Disturbance):
    r"""
    **A body-fixed thrust: a force at a point, with the torque it implies.**

    A thruster (or a leak, or an outgassing vent) pushes on the spacecraft
    with a force :math:`\mathbf{F}` applied at a body-frame point
    :math:`\mathbf{r}_t`.  That one physical quantity produces both a force on
    the centre of mass and a torque about it:

    .. math::

        \mathbf{F}_{\mathrm{thrust}} = \mathbf{F}_0 + \mathbf{n}(t),
        \qquad
        \mathbf{T}_{\mathrm{thrust}} =
        (\mathbf{r}_t - \mathbf{r}_{\mathrm{COM}}) \times \mathbf{F}_{\mathrm{thrust}},

    where :math:`\mathbf{F}_0` is the nominal thrust vector [N] and
    :math:`\mathbf{n}(t)` an optional additive noise drawn once per plant step
    by :meth:`update` (see :class:`~ADCS.satellite_hardware.errors.noise.Noise`).
    A thrust that acts through the centre of mass produces no torque.

    This differs from :class:`~ADCS.satellite_hardware.disturbances.prop_disturbance.Prop_Disturbance`,
    which models a propulsion *torque* directly with no force.  Use this class
    when the thrust itself matters (orbit changes, thrust misalignment torque
    that scales with the thrust level); use ``Prop_Disturbance`` when only a
    torque is known.

    Because the force is fixed in the body frame it does not depend on the
    attitude quaternion, so every attitude Jacobian and Hessian is zero.

    :param force_nominal: Nominal thrust vector in the body frame [N], shape ``(3,)``.
    :type force_nominal: numpy.ndarray
    :param location: Point of application in the body frame [m], shape ``(3,)``.
        ``None`` means the thrust acts through the centre of mass (no torque).
    :type location: numpy.ndarray | None
    :param noise: Noise model for the additive force noise [N]. Broadcast to three axes.
    :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` | None
    """

    def __init__(
        self,
        force_nominal: np.ndarray,
        location: Optional[np.ndarray] = None,
        noise: Optional[Noise] = None,
    ):
        self.force_nominal = np.asarray(force_nominal, dtype=float).reshape(3).copy()
        self.location = None if location is None else np.asarray(location, dtype=float).reshape(3).copy()
        self.noise = (noise if noise else Noise()).broadcast_to(3, owner="Thrust_Disturbance")
        self.current_force = self.force_nominal.copy()
        super().__init__()

    def update(self) -> None:
        r"""Draw one noise sample and hold it for the coming plant step.

        :math:`\mathbf{F}_{\mathrm{thrust}} = \mathbf{F}_0 + \mathbf{n}`.  A
        noise model with zero standard deviation keeps any fixed offset it
        carries without drawing.
        """
        std_noise = self.noise.std_noise
        if std_noise[0] > 0.0 or std_noise[1] > 0.0 or std_noise[2] > 0.0:
            self.noise._update_noise()
        self.current_force = self.force_nominal + self.noise.get_noise()

    def force(self, sat: "Satellite", x: State, os: Orbital_State) -> np.ndarray:
        r"""Current thrust force in the body frame [N], shape ``(3,)``.

        ``sat``, ``x`` and ``os`` are accepted for interface consistency and
        are unused: the thrust is body-fixed.
        """
        return self.current_force

    def torque(self, sat: "Satellite", x: State, os: Orbital_State) -> np.ndarray:
        r"""Torque of the thrust about the centre of mass [N·m], shape ``(3,)``.

        :math:`(\mathbf{r}_t - \mathbf{r}_{\mathrm{COM}}) \times \mathbf{F}_{\mathrm{thrust}}`,
        or zero when the thrust acts through the centre of mass.
        """
        if self.location is None:
            return np.zeros(3)
        return np.cross(self.location - sat.COM, self.current_force)

    def torque_qjac(self, *args, **kwargs) -> np.ndarray:
        """Zero: the thrust is body-fixed and does not depend on attitude. Shape ``(4, 3)``."""
        return np.zeros((4, 3))

    def torque_qqhess(self, *args, **kwargs) -> np.ndarray:
        """Zero: the thrust is body-fixed and does not depend on attitude. Shape ``(4, 4, 3)``."""
        return np.zeros((4, 4, 3))
