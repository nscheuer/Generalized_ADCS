"""Thin numeric dispatch boundary for compiled ADCS controllers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numba import njit

from .bcross import step as bcross_step
from .bdot import step as bdot_step
from .controller import CTRL_BCROSS, CTRL_BDOT, CTRL_NONE, Controller, NoController

if TYPE_CHECKING:
    from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
        SensorConfig,
    )

_CTRL_PARAM_WIDTH = 4


class ControllerTable:
    """Registry that flattens ``Controller`` objects into Numba-ready tables."""

    def __init__(self) -> None:
        self.kinds: list[int] = [CTRL_NONE]
        self.params: list[np.ndarray] = [np.zeros(_CTRL_PARAM_WIDTH, dtype=np.float64)]
        self.objects: list[Controller] = [NoController()]

    def register(self, controller: Controller | None) -> int:
        if controller is None or isinstance(controller, NoController):
            return 0
        if not isinstance(controller, Controller):
            raise TypeError(f"unsupported controller: {type(controller).__name__}")
        p = np.zeros(_CTRL_PARAM_WIDTH, dtype=np.float64)
        raw = controller.to_params()
        p[: raw.size] = raw
        idx = len(self.kinds)
        self.kinds.append(int(controller.kind))
        self.params.append(p)
        self.objects.append(controller)
        return idx


class ControllerRuntime(NamedTuple):
    """Flat numeric representation of registered controllers and magnetorquers."""

    enabled: bool
    kinds: np.ndarray
    params: np.ndarray
    mtq_axes: np.ndarray
    mtq_max_dipole: np.ndarray


def prepare_controllers(
    controllers: Sequence[Controller | None],
    sensors: Sequence[object],
    mtq_axes: np.ndarray,
    mtq_max_dipole: np.ndarray,
    table: ControllerTable | None = None,
) -> ControllerRuntime:
    """Validate sensor requirements and compile controller tables."""
    if table is None:
        table = ControllerTable()
    for ctrl in controllers:
        table.register(ctrl)
    for ctrl in table.objects:
        ctrl.validate_sensors(sensors)
    enabled = any(k != CTRL_NONE for k in table.kinds)
    return ControllerRuntime(
        enabled=enabled,
        kinds=np.asarray(table.kinds, dtype=np.int64),
        params=np.vstack(table.params),
        mtq_axes=np.ascontiguousarray(mtq_axes, dtype=np.float64),
        mtq_max_dipole=np.ascontiguousarray(mtq_max_dipole, dtype=np.float64),
    )


def init_controller_work() -> np.ndarray:
    """Allocate mutable controller state ``[prev_b(3), has_prev_b]``."""
    return np.zeros(4, dtype=np.float64)


@njit(cache=True)
def step_controller(
    ctrl_idx: int,
    dt_ctrl: float,
    readings: np.ndarray,
    sensor_config: SensorConfig,
    ctrl_rt: ControllerRuntime,
    ctrl_work: np.ndarray,
    dipole_commands: np.ndarray,
) -> None:
    """Dispatch to the compiled control law for ``ctrl_idx``."""
    kind = ctrl_rt.kinds[ctrl_idx]
    if kind == CTRL_NONE or dipole_commands.size == 0:
        for i in range(dipole_commands.size):
            dipole_commands[i] = 0.0
    elif kind == CTRL_BDOT:
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
    elif kind == CTRL_BCROSS:
        bcross_step(
            ctrl_rt.params[ctrl_idx, 0],
            readings,
            sensor_config,
            ctrl_rt.mtq_axes,
            ctrl_rt.mtq_max_dipole,
            dipole_commands,
        )
