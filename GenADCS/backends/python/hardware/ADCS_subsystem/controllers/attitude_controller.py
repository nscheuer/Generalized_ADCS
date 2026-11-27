"""Four-stage AttitudeController (feedforward, feedback, allocator, desaturation)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numba import njit

from .allocator import ALLOC_L2, Allocator, L2Allocator, allocate_l2, allocate_linfinity
from .controller import CTRL_ATTITUDE, Controller
from .desaturation import (
    DESAT_HOGAN_SCHAUB,
    Desaturation,
    NoDesaturation,
    hogan_schaub_mtq_dump,
    hogan_schaub_u0,
)
from .feedback import FB_MRP_FEEDBACK_LINEARIZED, Feedback, mrp_feedback_linearized_step
from .feedforward import FF_PASSTHROUGH, Feedforward, Passthrough, passthrough_step

if False:
    from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
        SensorConfig,
    )


@njit(cache=True)
def step(
    q: np.ndarray,
    omega: np.ndarray,
    q_goal: np.ndarray,
    w_goal: np.ndarray,
    wheel_momenta: np.ndarray,
    total_inertia: np.ndarray,
    bus_inertia: np.ndarray,
    wheel_axes: np.ndarray,
    wheel_max_torque: np.ndarray,
    a_pinv: np.ndarray,
    null_basis: np.ndarray,
    null_proj: np.ndarray,
    ff_kind: int,
    fb_kind: int,
    alloc_kind: int,
    desat_kind: int,
    kp: np.ndarray,
    kd: np.ndarray,
    desat_gain: float,
    desat_target: float,
    desat_mtq_gain: float,
    readings: np.ndarray,
    sensor_config: SensorConfig,
    mtq_axes: np.ndarray,
    mtq_max_dipole: np.ndarray,
    wheel_commands: np.ndarray,
    dipole_commands: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute the 4-stage AttitudeController in nopython mode.

    Returns ``(ff_torque, q_d, w_d, tau_c)`` for optional logging.
    """
    # 1. Feedforward
    if ff_kind == FF_PASSTHROUGH:
        q_d, w_d, w_d_dot, ff_torque = passthrough_step(q_goal, w_goal, q)
    else:
        q_d = q.copy()
        w_d = np.zeros(3, dtype=np.float64)
        w_d_dot = np.zeros(3, dtype=np.float64)
        ff_torque = np.zeros(3, dtype=np.float64)

    # 2. Feedback
    if fb_kind == FB_MRP_FEEDBACK_LINEARIZED:
        tau_c = mrp_feedback_linearized_step(
            q,
            omega,
            q_d,
            w_d,
            w_d_dot,
            ff_torque,
            wheel_momenta,
            total_inertia,
            bus_inertia,
            wheel_axes,
            kp,
            kd,
        )
    else:
        tau_c = ff_torque.copy()

    if wheel_commands.size == 0:
        return ff_torque, q_d, w_d, tau_c

    # 3. Desaturation nullspace vector u_0 + optional magnetorquer momentum dump
    if desat_kind == DESAT_HOGAN_SCHAUB:
        u_0 = hogan_schaub_u0(wheel_momenta, desat_gain, desat_target)
        hogan_schaub_mtq_dump(
            wheel_momenta,
            wheel_axes,
            desat_mtq_gain,
            desat_target,
            readings,
            sensor_config,
            mtq_axes,
            mtq_max_dipole,
            dipole_commands,
        )
    else:
        u_0 = np.zeros(wheel_commands.size, dtype=np.float64)
        for i in range(dipole_commands.size):
            dipole_commands[i] = 0.0

    # 4. Allocator
    if alloc_kind == ALLOC_L2:
        u_rw = allocate_l2(tau_c, a_pinv, null_proj, u_0, wheel_max_torque)
    else:
        u_rw = allocate_linfinity(
            tau_c, a_pinv, null_basis, null_proj, u_0, wheel_max_torque
        )

    for i in range(wheel_commands.size):
        wheel_commands[i] = u_rw[i]

    return ff_torque, q_d, w_d, tau_c


@dataclass(slots=True)
class AttitudeController(Controller):
    """Four-stage reaction-wheel attitude controller.

    Combines ``feedforward``, ``feedback``, ``allocator``, and ``desaturation``.
    """

    feedback: Feedback
    feedforward: Feedforward = field(default_factory=Passthrough)
    allocator: Allocator = field(default_factory=L2Allocator)
    desaturation: Desaturation = field(default_factory=NoDesaturation)
    kind: ClassVar[int] = CTRL_ATTITUDE
    kernel = staticmethod(step)

    def __post_init__(self) -> None:
        if not isinstance(self.feedback, Feedback):
            raise TypeError("feedback must be a Feedback instance")
        if not isinstance(self.feedforward, Feedforward):
            raise TypeError("feedforward must be a Feedforward instance")
        if not isinstance(self.allocator, Allocator):
            raise TypeError("allocator must be an Allocator instance")
        if not isinstance(self.desaturation, Desaturation):
            raise TypeError("desaturation must be a Desaturation instance")

    def validate_sensors(self, sensors: Sequence[object]) -> None:
        return None

    def to_params(self) -> np.ndarray:
        desat_p = self.desaturation.to_params()
        return np.array(
            [
                float(self.feedforward.kind),
                float(self.feedback.kind),
                float(self.allocator.kind),
                float(self.desaturation.kind),
                desat_p[0],
                desat_p[1],
                desat_p[2],
            ],
            dtype=np.float64,
        )
