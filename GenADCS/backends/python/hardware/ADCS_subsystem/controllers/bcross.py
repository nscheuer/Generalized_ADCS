"""B-cross magnetic rate-damping controller."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numba import njit

from .controller import CTRL_BCROSS, Controller, allocate_dipoles, read_first_sensor_body

if TYPE_CHECKING:
    from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
        SensorConfig,
    )


@njit(cache=True)
def step(
    gain: float,
    readings: np.ndarray,
    sensor_config: SensorConfig,
    mtq_axes: np.ndarray,
    mtq_max_dipole: np.ndarray,
    dipole_commands: np.ndarray,
) -> None:
    """Compute B-cross magnetorquer commands: ``m = gain * (omega x B) / ||B||^2``."""
    ok_mag, b_body = read_first_sensor_body(1, readings, sensor_config)
    ok_gyro, w_body = read_first_sensor_body(0, readings, sensor_config)
    if not (ok_mag and ok_gyro):
        for i in range(dipole_commands.size):
            dipole_commands[i] = 0.0
        return
    b_sq = np.dot(b_body, b_body)
    if b_sq <= 1e-18:
        for i in range(dipole_commands.size):
            dipole_commands[i] = 0.0
        return
    dipole_body = (gain / b_sq) * np.cross(w_body, b_body)
    allocate_dipoles(dipole_body, mtq_axes, mtq_max_dipole, dipole_commands)


@dataclass(slots=True)
class BCross(Controller):
    """B-cross magnetic rate-damping control law: ``m = gain * (omega x B) / ||B||^2``.

    Requires both a ``Magnetometer`` and a ``Gyroscope`` onboard ``satellite.adcs.sensors``.
    """

    gain: float
    kind: ClassVar[int] = CTRL_BCROSS
    kernel = staticmethod(step)

    def __post_init__(self) -> None:
        self.gain = float(self.gain)
        if not math.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError("BCross gain must be finite and positive")

    def validate_sensors(self, sensors: Sequence[object]) -> None:
        names = {type(s).__name__ for s in sensors}
        if "Magnetometer" not in names or "Gyroscope" not in names:
            raise ValueError(
                "BCross controller requires both a Magnetometer and a Gyroscope onboard the satellite"
            )

    def to_params(self) -> np.ndarray:
        return np.array([self.gain], dtype=np.float64)
