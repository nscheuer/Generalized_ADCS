"""Unbounded UTC playback and precomputed simulation transforms."""

from datetime import datetime, timezone

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PlaybackClock:
    """A UTC clock that keeps advancing beyond recorded simulation intervals."""

    SPEEDS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)

    def __init__(self, time_hist=None, start_utc: datetime | None = None) -> None:
        self.time = np.asarray([] if time_hist is None else time_hist, dtype=float).reshape(-1)
        if not np.all(np.isfinite(self.time)) or np.any(np.diff(self.time) <= 0):
            raise ValueError("time_hist must contain finite, strictly increasing times")
        if start_utc is None:
            start_utc = datetime.now(timezone.utc)
        elif start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=timezone.utc)
        else:
            start_utc = start_utc.astimezone(timezone.utc)
        self.start_utc = start_utc
        self.current = start_utc.timestamp()
        self.playing = True
        self.speed = 1.0

    @property
    def data_interval(self) -> tuple[float, float] | None:
        if not len(self.time):
            return None
        return self.start_utc.timestamp(), self.start_utc.timestamp() + float(self.time[-1] - self.time[0])

    def simulation_time(self, utc_timestamp: float) -> float | None:
        if not len(self.time):
            return None
        return float(self.time[0] + utc_timestamp - self.start_utc.timestamp())

    def seek(self, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("UTC time must be finite")
        self.current = float(value)

    def advance(self, dt: float) -> None:
        if not self.playing:
            return
        self.current += max(0.0, dt) * self.speed

    def toggle(self) -> None:
        self.playing = not self.playing

    def change_speed(self, direction: int) -> None:
        index = min(range(len(self.SPEEDS)), key=lambda i: abs(self.SPEEDS[i] - self.speed))
        index = max(0, min(len(self.SPEEDS) - 1, index + direction))
        self.speed = self.SPEEDS[index]


class EarthOrientationTrack:
    """Convert Earth-fixed axes to ECI once per recorded orbital state."""

    def __init__(self, time_hist, os_hist) -> None:
        if os_hist is None or len(os_hist) == 0:
            self._constant = Rotation.identity()
            self._slerp = None
            return

        basis = np.eye(3)
        matrices = np.asarray([
            np.column_stack([orb.ecef_to_eci(axis) for axis in basis])
            for orb in os_hist
        ])
        rotations = Rotation.from_matrix(matrices)
        self._constant = rotations[0]
        self._first_time = float(time_hist[0])
        self._last_time = float(time_hist[-1])
        self._slerp = Slerp(time_hist, rotations) if len(time_hist) > 1 else None

    def panda_quat(self, time: float | None):
        from panda3d.core import Quat

        if self._slerp is not None and time is not None:
            rotation = self._slerp([np.clip(time, self._first_time, self._last_time)])[0]
        else:
            rotation = self._constant
        x, y, z, w = rotation.as_quat()
        return Quat(w, x, y, z)
