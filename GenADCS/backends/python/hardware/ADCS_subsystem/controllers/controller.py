"""Base contract and shared Numba helpers for compiled ADCS controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numba import njit

from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    quaternion_to_dcm,
)

if TYPE_CHECKING:
    from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
        SensorConfig,
    )

CTRL_NONE = 0
CTRL_BDOT = 1
CTRL_BCROSS = 2


@njit(cache=True, inline="always")
def read_first_sensor_body(
    target_code: int,
    readings: np.ndarray,
    sensor_config: SensorConfig,
) -> tuple[bool, np.ndarray]:
    """Read first enabled sensor of ``target_code`` (0=gyro, 1=mag) in body frame."""
    code = sensor_config[0]
    orientation = sensor_config[1]
    enabled = sensor_config[8]
    offsets = sensor_config[9]
    for i in range(code.size):
        if code[i] == target_code and enabled[i]:
            off = offsets[i]
            raw = readings[off : off + 3]
            if not (np.isnan(raw[0]) or np.isnan(raw[1]) or np.isnan(raw[2])):
                dcm = quaternion_to_dcm(orientation[i])
                return True, dcm.T @ raw
    return False, np.zeros(3, dtype=np.float64)


@njit(cache=True, inline="always")
def allocate_dipoles(
    dipole_body: np.ndarray,
    mtq_axes: np.ndarray,
    mtq_max_dipole: np.ndarray,
    dipole_commands: np.ndarray,
) -> None:
    """Project desired 3D dipole onto magnetorquer axes and saturate."""
    for i in range(dipole_commands.size):
        cmd = np.dot(dipole_body, mtq_axes[i])
        limit = mtq_max_dipole[i]
        if cmd > limit:
            cmd = limit
        elif cmd < -limit:
            cmd = -limit
        dipole_commands[i] = cmd


class Controller(ABC):
    """Base class for ADCS control laws compiled to Numba."""

    kind: ClassVar[int]

    @abstractmethod
    def validate_sensors(self, sensors: Sequence[object]) -> None:
        """Raise ``ValueError`` if required onboard sensors are missing."""

    @abstractmethod
    def to_params(self) -> np.ndarray:
        """Return fixed-width ``float64`` parameter vector for the compiled kernel."""


class NoController(Controller):
    """Open-loop controller outputting zero actuator commands."""

    kind: ClassVar[int] = CTRL_NONE

    def validate_sensors(self, sensors: Sequence[object]) -> None:
        return None

    def to_params(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float64)
