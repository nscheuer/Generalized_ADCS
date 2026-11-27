"""Hogan & Schaub continuous nullspace + magnetorquer reaction-wheel momentum management."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numba import njit

from ..controller import allocate_dipoles, read_first_sensor_body
from .desaturation import DESAT_HOGAN_SCHAUB, Desaturation

if TYPE_CHECKING:
    from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
        SensorConfig,
    )


@njit(cache=True, inline="always")
def hogan_schaub_u0(
    wheel_momenta: np.ndarray,
    gain: float,
    target_momentum: float,
) -> np.ndarray:
    """Continuous nullspace momentum management: ``u_0 = +k * (h - h*)``."""
    return gain * (wheel_momenta - target_momentum)


@njit(cache=True)
def hogan_schaub_mtq_dump(
    wheel_momenta: np.ndarray,
    wheel_axes: np.ndarray,
    mtq_gain: float,
    target_momentum: float,
    readings: np.ndarray,
    sensor_config: SensorConfig,
    mtq_axes: np.ndarray,
    mtq_max_dipole: np.ndarray,
    dipole_commands: np.ndarray,
) -> None:
    """Dump net wheel momentum error via magnetorquers: ``m = k_m (H_err x B) / ||B||^2``."""
    if mtq_gain <= 0.0 or dipole_commands.size == 0:
        for i in range(dipole_commands.size):
            dipole_commands[i] = 0.0
        return
    ok_mag, b_body = read_first_sensor_body(1, readings, sensor_config)
    if not ok_mag:
        for i in range(dipole_commands.size):
            dipole_commands[i] = 0.0
        return
    b_sq = np.dot(b_body, b_body)
    if b_sq <= 1e-18:
        for i in range(dipole_commands.size):
            dipole_commands[i] = 0.0
        return
    h_err_body = np.zeros(3, dtype=np.float64)
    for w in range(wheel_momenta.size):
        h_err_body += wheel_axes[w] * (wheel_momenta[w] - target_momentum)
    dipole_body = (mtq_gain / b_sq) * np.cross(h_err_body, b_body)
    allocate_dipoles(dipole_body, mtq_axes, mtq_max_dipole, dipole_commands)


@dataclass(slots=True)
class HoganSchaubManagement(Desaturation):
    """Continuous nullspace + optional magnetorquer wheel momentum management."""

    gain: float
    target_momentum: float = 0.0
    mtq_gain: float = 0.0
    kind: ClassVar[int] = DESAT_HOGAN_SCHAUB

    def __post_init__(self) -> None:
        self.gain = float(self.gain)
        self.target_momentum = float(self.target_momentum)
        self.mtq_gain = float(self.mtq_gain)
        if not math.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError("HoganSchaubManagement gain must be finite and positive")
        if not math.isfinite(self.target_momentum):
            raise ValueError("target_momentum must be finite")
        if not math.isfinite(self.mtq_gain) or self.mtq_gain < 0.0:
            raise ValueError("mtq_gain must be finite and nonnegative")

    def to_params(self) -> np.ndarray:
        return np.array([self.gain, self.target_momentum, self.mtq_gain], dtype=np.float64)
