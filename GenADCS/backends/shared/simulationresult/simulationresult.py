"""Backend-independent, numeric simulation output and HDF5 persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re

import numpy as np

from .satellite_configuration import to_satellite as rebuild_satellite


@dataclass(slots=True)
class SatelliteResult:
    """Samples for one satellite at elapsed times in seconds."""

    times_s: np.ndarray
    attitude_quaternion: np.ndarray | None
    satellite_rate: np.ndarray | None
    wheel_momenta: np.ndarray | None
    satellite_config: dict
    _backend: str = field(default="", init=False, repr=False)

    def to_satellite(self):
        """Reconstruct the satellite hardware from its saved configuration."""
        return rebuild_satellite(self.satellite_config, self._backend)

    def __post_init__(self) -> None:
        self.times_s = np.asarray(self.times_s, dtype=np.float64)
        if self.times_s.ndim != 1 or not len(self.times_s) or not np.all(np.isfinite(self.times_s)):
            raise ValueError("times_s must be a nonempty, finite 1D array")
        if np.any(np.diff(self.times_s) < 0):
            raise ValueError("times_s must be ordered")
        for name, width in (("attitude_quaternion", 4), ("satellite_rate", 3), ("wheel_momenta", None)):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 2 or array.shape[0] != len(self.times_s) or (width is not None and array.shape[1] != width):
                raise ValueError(f"{name} must have one row per sample and {width or 'a fixed number of'} columns")
            setattr(self, name, array)


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
            file.attrs["schema_version"] = 4
            file.attrs["backend"] = self.backend
            file.attrs["N"] = self.N
            file.attrs["epoch_jd"] = self.epoch_jd
            file.attrs["sampling"] = "initial and every Nth full dt interval"
            for key, satellite in self.satellites.items():
                group = file.create_group(key)
                group.attrs["satellite_config"] = json.dumps(satellite.satellite_config)
                group.create_dataset("times_s", data=satellite.times_s)
                for name in ("attitude_quaternion", "satellite_rate", "wheel_momenta"):
                    value = getattr(satellite, name)
                    if value is not None:
                        group.create_dataset(name, data=value)
        return path

    @classmethod
    def load(cls, path: str | Path) -> SimulationResult:
        """Read run metadata and all satellite groups."""
        import h5py

        with h5py.File(path, "r") as file:
            if file.attrs.get("schema_version") != 4:
                raise ValueError("Unsupported SimulationResult HDF5 schema version")
            satellites = {}
            for key, group in file.items():
                fields = {name: group[name][...] if name in group else None for name in
                          ("attitude_quaternion", "satellite_rate", "wheel_momenta")}
                satellites[key] = SatelliteResult(group["times_s"][...], **fields,
                                                  satellite_config=json.loads(group.attrs["satellite_config"]))
            return cls(epoch_jd=float(file.attrs["epoch_jd"]), N=int(file.attrs["N"]),
                       backend=str(file.attrs["backend"]), satellites=satellites)
