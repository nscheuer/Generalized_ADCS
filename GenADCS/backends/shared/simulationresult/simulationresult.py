"""Backend-independent, numeric simulation output and HDF5 persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re

import numpy as np

from .satellite_configuration import to_satellite as rebuild_satellite

RESULT_FIELDS = (
    "satellite_attitude", "satellite_rate", "satellite_wheel_momenta",
    "estimated_attitude", "estimated_rate", "estimated_covariance",
    "estimated_attitude_covariance", "estimated_rate_covariance",
    "measurement_innovation", "measurement_innovation_covariance",
    "normalized_innovation_squared", "normalized_estimation_error_squared",
    "measurement_residual", "kalman_gain",
    "satellite_position", "satellite_velocity", "relative_sun_vector",
    "relative_earth_vector", "atmospheric_density", "relative_atmospheric_velocity",
    "magnetic_field", "eclipse_state", "gravity_force", "aerodynamic_force",
    "solar_radiation_force", "J2_force", "J3_force", "J4_force",
    "J5_force", "J6_force", "lunisolar_force",
    "gravity_gradient_torque", "aerodynamic_torque", "solar_radiation_torque",
    "magnetic_torque", "total_force", "total_torque",
    "gyroscope_clean", "gyroscope_reading", "magnetometer_clean", "magnetometer_reading",
    "autonomous_star_tracker_clean", "autonomous_star_tracker_reading",
    "coarse_sun_sensor_clean", "coarse_sun_sensor_reading",
    "fine_sun_sensor_clean", "fine_sun_sensor_reading",
    "fine_earth_sensor_clean", "fine_earth_sensor_reading",
    "gnss_clean", "gnss_reading", "accelerometer_clean", "accelerometer_reading",
    "obc_mode", "adcs_mode", "adcs_goal",
)

SENSOR_FIELDS = frozenset(name for name in RESULT_FIELDS if name.endswith(("_clean", "_reading")))


@dataclass(slots=True)
class SatelliteResult:
    """Samples for one satellite at elapsed times in seconds."""

    times_s: np.ndarray
    fields: dict[str, np.ndarray]
    satellite_config: dict
    _backend: str = field(default="", init=False, repr=False)

    def to_satellite(self):
        """Reconstruct the satellite hardware from its saved configuration."""
        return rebuild_satellite(self.satellite_config, self._backend)

    def __getattr__(self, name):
        if name in RESULT_FIELDS:
            return self.fields.get(name)
        raise AttributeError(name)

    def __post_init__(self) -> None:
        self.times_s = np.asarray(self.times_s, dtype=np.float64)
        if self.times_s.ndim != 1 or not len(self.times_s) or not np.all(np.isfinite(self.times_s)):
            raise ValueError("times_s must be a nonempty, finite 1D array")
        if np.any(np.diff(self.times_s) < 0):
            raise ValueError("times_s must be ordered")
        for name, value in self.fields.items():
            if name not in RESULT_FIELDS:
                raise ValueError(f"unknown result field: {name}")
            width = (None if name in SENSOR_FIELDS else 1 if name in ("atmospheric_density", "eclipse_state", "normalized_innovation_squared", "normalized_estimation_error_squared", "obc_mode", "adcs_mode", "adcs_goal")
                     else None if name == "satellite_wheel_momenta" else 4 if name in ("satellite_attitude", "estimated_attitude")
                     else (36, 49) if name == "estimated_covariance" else 18 if name == "kalman_gain"
                     else 9 if name in ("estimated_attitude_covariance", "estimated_rate_covariance", "measurement_innovation_covariance") else 3)
            array = np.asarray(value, dtype=np.float64)
            valid_width = width is None or (array.shape[1] in width if isinstance(width, tuple) else array.shape[1] == width)
            if array.ndim != 2 or array.shape[0] != len(self.times_s) or not valid_width:
                raise ValueError(f"{name} must have one row per sample and {width or 'a fixed number of'} columns")
            self.fields[name] = array


@dataclass(slots=True)
class SimulationResult:
    """Run metadata and samples indexed by satellite IDs such as ``sat_000``.

    The epoch is a Julian date stored once for the run. Each satellite's
    sample times are elapsed seconds from that epoch. Samples are taken at
    step zero and every ``N`` full ``dt`` intervals.
    """

    epoch_jd: float
    N: int
    backend: str
    satellites: dict[str, SatelliteResult]

    def __post_init__(self) -> None:
        self.epoch_jd = float(self.epoch_jd)
        if not np.isfinite(self.epoch_jd):
            raise ValueError("epoch_jd must be finite")
        if isinstance(self.N, bool) or not isinstance(self.N, int) or self.N < 1:
            raise ValueError("N must be a positive integer")
        if not isinstance(self.satellites, dict) or not self.satellites:
            raise ValueError("satellites must be a nonempty dictionary")
        for key, value in self.satellites.items():
            if not isinstance(key, str) or re.fullmatch(r"sat_\d{3,}", key) is None:
                raise ValueError("satellite keys must have the form sat_000")
            if not isinstance(value, SatelliteResult):
                raise TypeError("satellite values must be SatelliteResult instances")
            value._backend = self.backend

    def save(self, path: str | Path) -> Path:
        """Write one HDF5 group per satellite."""
        import h5py

        path = Path(path)
        with h5py.File(path, "w") as file:
            file.attrs["schema_version"] = 5
            file.attrs["backend"] = self.backend
            file.attrs["N"] = self.N
            file.attrs["epoch_jd"] = self.epoch_jd
            file.attrs["sampling"] = "initial and every Nth full dt interval"
            for key, satellite in self.satellites.items():
                group = file.create_group(key)
                group.attrs["satellite_config"] = json.dumps(satellite.satellite_config)
                group.create_dataset("times_s", data=satellite.times_s)
                for name, value in satellite.fields.items():
                    group.create_dataset(name, data=value)
        return path

    @classmethod
    def load(cls, path: str | Path) -> SimulationResult:
        """Read run metadata and all satellite groups."""
        import h5py

        with h5py.File(path, "r") as file:
            if file.attrs.get("schema_version") != 5:
                raise ValueError("Unsupported SimulationResult HDF5 schema version")
            satellites = {}
            for key, group in file.items():
                fields = {name: group[name][...] for name in group if name != "times_s"}
                satellites[key] = SatelliteResult(group["times_s"][...], fields,
                                                  json.loads(group.attrs["satellite_config"]))
            return cls(epoch_jd=float(file.attrs["epoch_jd"]), N=int(file.attrs["N"]),
                       backend=str(file.attrs["backend"]), satellites=satellites)
