"""B-dot magnetic detumbling controller."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numba import njit

from .controller import CTRL_BDOT, Controller, allocate_dipoles, read_first_sensor_body

if TYPE_CHECKING:
    from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
        SensorConfig,
    )


@njit(cache=True)
def step(
    gain: float,
    dt_ctrl: float,
    readings: np.ndarray,
    sensor_config: SensorConfig,
    mtq_axes: np.ndarray,
    mtq_max_dipole: np.ndarray,
    ctrl_work: np.ndarray,
    dipole_commands: np.ndarray,
) -> None:
    """Compute B-dot magnetorquer commands: ``m = -gain * dB_body / dt``."""
    ok_mag, b_body = read_first_sensor_body(1, readings, sensor_config)
    if not ok_mag:
        for i in range(dipole_commands.size):
            dipole_commands[i] = 0.0
        return
    if ctrl_work[3] == 0.0 or dt_ctrl <= 0.0:
        ctrl_work[0:3] = b_body
        ctrl_work[3] = 1.0
        for i in range(dipole_commands.size):
            dipole_commands[i] = 0.0
        return
    b_dot = (b_body - ctrl_work[0:3]) / dt_ctrl
    ctrl_work[0:3] = b_body
    dipole_body = -gain * b_dot
    allocate_dipoles(dipole_body, mtq_axes, mtq_max_dipole, dipole_commands)


@dataclass(slots=True)
class BDot(Controller):
    """B-dot detumbling control law: ``m = -gain * dB_body / dt``.

    Requires at least one ``Magnetometer`` onboard ``satellite.adcs.sensors``.
    """

    gain: float
    kind: ClassVar[int] = CTRL_BDOT
    kernel = staticmethod(step)

    def __post_init__(self) -> None:
        self.gain = float(self.gain)
        if not math.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError("BDot gain must be finite and positive")

    def validate_sensors(self, sensors: Sequence[object]) -> None:
        has_mag = any(type(s).__name__ == "Magnetometer" for s in sensors)
        if not has_mag:
            raise ValueError("BDot controller requires a Magnetometer onboard the satellite")

    def to_params(self) -> np.ndarray:
        return np.array([self.gain], dtype=np.float64)
