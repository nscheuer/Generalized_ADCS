"""Thin numeric dispatch boundary for compiled ADCS controllers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numba import njit

from .attitude_controller import AttitudeController
from .attitude_controller import step as attitude_step
from .bcross import step as bcross_step
from .bdot import step as bdot_step
from .controller import (
    CTRL_ATTITUDE,
    CTRL_BCROSS,
    CTRL_BDOT,
    CTRL_NONE,
    Controller,
    NoController,
)

if TYPE_CHECKING:
    from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
        SensorConfig,
    )

_CTRL_PARAM_WIDTH = 6


class ControllerTable:
    """Registry that flattens ``Controller`` objects into Numba-ready tables."""

    def __init__(self) -> None:
        self.kinds: list[int] = [CTRL_NONE]
        self.params: list[np.ndarray] = [np.zeros(_CTRL_PARAM_WIDTH, dtype=np.float64)]
        self.kp: list[np.ndarray] = [np.zeros((3, 3), dtype=np.float64)]
        self.kd: list[np.ndarray] = [np.zeros((3, 3), dtype=np.float64)]
        self.objects: list[Controller] = [NoController()]

    def register(self, controller: Controller | None) -> int:
        if controller is None or isinstance(controller, NoController):
            return 0
        if not isinstance(controller, Controller):
            raise TypeError(f"unsupported controller: {type(controller).__name__}")
        p = np.zeros(_CTRL_PARAM_WIDTH, dtype=np.float64)
        raw = controller.to_params()
        p[: raw.size] = raw
        if isinstance(controller, AttitudeController):
            kp_mat, kd_mat = controller.feedback.to_matrices()
        else:
            kp_mat = np.zeros((3, 3), dtype=np.float64)
            kd_mat = np.zeros((3, 3), dtype=np.float64)
        idx = len(self.kinds)
        self.kinds.append(int(controller.kind))
        self.params.append(p)
        self.kp.append(kp_mat)
        self.kd.append(kd_mat)
        self.objects.append(controller)
        return idx


class ControllerRuntime(NamedTuple):
    """Flat numeric representation of registered controllers and actuators."""

    enabled: bool
    kinds: np.ndarray
    params: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    mtq_axes: np.ndarray
    mtq_max_dipole: np.ndarray
    wheel_axes: np.ndarray
    wheel_max_torque: np.ndarray
    a_pinv: np.ndarray
    null_basis: np.ndarray
    null_proj: np.ndarray
    total_inertia: np.ndarray
    bus_inertia: np.ndarray


def prepare_controllers(
    controllers: Sequence[Controller | None],
    sensors: Sequence[object],
    mtq_axes: np.ndarray,
    mtq_max_dipole: np.ndarray,
    table: ControllerTable | None = None,
    wheel_axes: np.ndarray | None = None,
    wheel_max_torque: np.ndarray | None = None,
    total_inertia: np.ndarray | None = None,
    bus_inertia: np.ndarray | None = None,
) -> ControllerRuntime:
    """Validate sensor requirements, precompute allocation matrices, and compile tables."""
    if table is None:
        table = ControllerTable()
    for ctrl in controllers:
        table.register(ctrl)
    for ctrl in table.objects:
        ctrl.validate_sensors(sensors)
    enabled = any(k != CTRL_NONE for k in table.kinds)

    w_axes = (
        np.ascontiguousarray(wheel_axes, dtype=np.float64)
        if wheel_axes is not None
        else np.zeros((0, 3), dtype=np.float64)
    )
    w_max_t = (
        np.ascontiguousarray(wheel_max_torque, dtype=np.float64)
        if wheel_max_torque is not None
        else np.zeros(0, dtype=np.float64)
    )
    n_w = w_axes.shape[0]
    if n_w > 0:
        a_mat = w_axes.T  # (3, n_w)
        a_pinv = np.ascontiguousarray(np.linalg.pinv(a_mat), dtype=np.float64)
        null_proj = np.ascontiguousarray(np.eye(n_w) - a_pinv @ a_mat, dtype=np.float64)
        _, s, vh = np.linalg.svd(a_mat, full_matrices=True)
        rank = int(np.sum(s > 1e-12))
        null_basis = np.ascontiguousarray(vh[rank:].T, dtype=np.float64)
    else:
        a_pinv = np.zeros((0, 3), dtype=np.float64)
        null_proj = np.zeros((0, 0), dtype=np.float64)
        null_basis = np.zeros((0, 0), dtype=np.float64)

    t_iner = (
        np.ascontiguousarray(total_inertia, dtype=np.float64)
        if total_inertia is not None
        else np.eye(3, dtype=np.float64)
    )
    b_iner = (
        np.ascontiguousarray(bus_inertia, dtype=np.float64)
        if bus_inertia is not None
        else t_iner
    )

    return ControllerRuntime(
        enabled=enabled,
        kinds=np.asarray(table.kinds, dtype=np.int64),
        params=np.vstack(table.params),
        kp=np.stack(table.kp),
        kd=np.stack(table.kd),
        mtq_axes=np.ascontiguousarray(mtq_axes, dtype=np.float64),
        mtq_max_dipole=np.ascontiguousarray(mtq_max_dipole, dtype=np.float64),
        wheel_axes=w_axes,
        wheel_max_torque=w_max_t,
        a_pinv=a_pinv,
        null_basis=null_basis,
        null_proj=null_proj,
        total_inertia=t_iner,
        bus_inertia=b_iner,
    )


def init_controller_work() -> np.ndarray:
    """Allocate mutable controller state ``[prev_b(3), has_prev_b]``."""
    return np.zeros(4, dtype=np.float64)


@njit(cache=True)
def step_controller(
    ctrl_idx: int,
    dt_ctrl: float,
    q: np.ndarray,
    omega: np.ndarray,
    q_goal: np.ndarray,
    w_goal: np.ndarray,
    wheel_momenta: np.ndarray,
    readings: np.ndarray,
    sensor_config: SensorConfig,
    ctrl_rt: ControllerRuntime,
    ctrl_work: np.ndarray,
    wheel_commands: np.ndarray,
    dipole_commands: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the compiled control law for ``ctrl_idx``."""
    kind = ctrl_rt.kinds[ctrl_idx]
    nan_q = np.full(4, np.nan, dtype=np.float64)
    nan_3 = np.full(3, np.nan, dtype=np.float64)

    if kind == CTRL_NONE:
        for i in range(dipole_commands.size):
            dipole_commands[i] = 0.0
        for i in range(wheel_commands.size):
            wheel_commands[i] = 0.0
        return nan_3, nan_q, nan_3, nan_3

    if kind == CTRL_BDOT:
        for i in range(wheel_commands.size):
            wheel_commands[i] = 0.0
        if dipole_commands.size > 0:
            bdot_step(
                ctrl_rt.params[ctrl_idx, 0],
                dt_ctrl,
                readings,
                sensor_config,
                ctrl_rt.mtq_axes,
                ctrl_rt.mtq_max_dipole,
                ctrl_work,
                dipole_commands,
            )
        return nan_3, nan_q, nan_3, nan_3

    if kind == CTRL_BCROSS:
        for i in range(wheel_commands.size):
            wheel_commands[i] = 0.0
        if dipole_commands.size > 0:
            bcross_step(
                ctrl_rt.params[ctrl_idx, 0],
                readings,
                sensor_config,
                ctrl_rt.mtq_axes,
                ctrl_rt.mtq_max_dipole,
                dipole_commands,
            )
        return nan_3, nan_q, nan_3, nan_3

    if kind == CTRL_ATTITUDE:
        for i in range(dipole_commands.size):
            dipole_commands[i] = 0.0
        return attitude_step(
            q,
            omega,
            q_goal,
            w_goal,
            wheel_momenta,
            ctrl_rt.total_inertia,
            ctrl_rt.bus_inertia,
            ctrl_rt.wheel_axes,
            ctrl_rt.wheel_max_torque,
            ctrl_rt.a_pinv,
            ctrl_rt.null_basis,
            ctrl_rt.null_proj,
            int(ctrl_rt.params[ctrl_idx, 0]),
            int(ctrl_rt.params[ctrl_idx, 1]),
            int(ctrl_rt.params[ctrl_idx, 2]),
            int(ctrl_rt.params[ctrl_idx, 3]),
            ctrl_rt.kp[ctrl_idx],
            ctrl_rt.kd[ctrl_idx],
            ctrl_rt.params[ctrl_idx, 4],
            ctrl_rt.params[ctrl_idx, 5],
            wheel_commands,
        )

    return nan_3, nan_q, nan_3, nan_3
