from __future__ import annotations
import pickle
import lzma
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterator, Union

from ADCS.satellite_hardware.satellite import Satellite, EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State

@dataclass
class RunResults:
    satellite: Satellite
    est_satellite: Optional[EstimatedSatellite] = None
    time_J2000: Optional[np.ndarray] = None
    time_s: Optional[np.ndarray] = None
    os_hist: Optional[List[Orbital_State]] = None
    est_os_hist: Optional[List[Orbital_State]] = None
    os_cov_hist: Optional[List[np.ndarray]] = None
    state_hist: Optional[np.ndarray] = None
    est_state_hist: Optional[np.ndarray] = None
    state_cov_hist: Optional[List[np.ndarray]] = None
    sensor_bias: Optional[np.ndarray] = None
    est_sensor_bias: Optional[np.ndarray] = None
    actuator_bias: Optional[np.ndarray] = None
    est_actuator_bias: Optional[np.ndarray] = None
    target_hist: Optional[np.ndarray] = None
    w_target_hist: Optional[np.ndarray] = None
    clean_sensor_hist: Optional[np.ndarray] = None
    sensor_hist: Optional[np.ndarray] = None
    control_hist: Optional[np.ndarray] = None

    def record(self, *, k: int, **kwargs):
        mapping = {
            "time_J2000": "time_J2000", "time_s": "time_s", "os": "os_hist",
            "est_os": "est_os_hist", "os_cov": "os_cov_hist", "state": "state_hist",
            "est_state": "est_state_hist", "state_cov": "state_cov_hist",
            "sensor_bias": "sensor_bias", "est_sensor_bias": "est_sensor_bias",
            "actuator_bias": "actuator_bias", "est_actuator_bias": "est_actuator_bias",
            "target": "target_hist", "w_target": "w_target_hist",
            "clean_sensor": "clean_sensor_hist", "sensor": "sensor_hist", "control": "control_hist"
        }
        for key, val in kwargs.items():
            if val is not None and key in mapping:
                attr = mapping[key]
                if getattr(self, attr) is None:
                    setattr(self, attr, [])
                if key in ["state", "est_state", "target", "w_target", "clean_sensor", "sensor", "control"]:
                    getattr(self, attr).append(np.asarray(val).copy())
                else:
                    getattr(self, attr).append(val)

    def flatten(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        if data.get("os_hist"):
            data["os_hist"] = [os.to_dict() if hasattr(os, "to_dict") else os for os in data["os_hist"]]
        if data.get("est_os_hist"):
            data["est_os_hist"] = [os.to_dict() if hasattr(os, "to_dict") else os for os in data["est_os_hist"]]
        return data

    @classmethod
    def inflate(cls, data: Dict[str, Any], ephem: Any = None) -> RunResults:
        if ephem is not None:
            if data.get("os_hist"):
                data["os_hist"] = [Orbital_State.from_dict(d, ephem=ephem) for d in data["os_hist"]]
            if data.get("est_os_hist"):
                data["est_os_hist"] = [Orbital_State.from_dict(d, ephem=ephem) for d in data["est_os_hist"]]
        return cls(**data)

@dataclass
class SimulationResults:
    runs: List[RunResults]
    configs: Optional[List[Dict[str, Any]]] = None
    run_ids: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.runs, list) or len(self.runs) == 0:
            raise ValueError("runs must be a non-empty list")

    def save(self, name: str, out_dir: str | Path = "output", compress: bool = True) -> Path:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sim"
        
        serializable_data = {
            "runs": [r.flatten() for r in self.runs],
            "configs": self.configs,
            "run_ids": self.run_ids
        }
        
        print(f"[SimulationResults] Saving sanitized data to {file_path}...")
        open_func = lzma.open if compress else open
        with open_func(file_path, "wb") as f:
            pickle.dump(serializable_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return file_path

    @classmethod
    def load(cls, path: str | Path, ephem: Any = None) -> SimulationResults:
        path = Path(path)
        open_func = lzma.open if path.suffix == ".sim" else open
        
        print(f"[SimulationResults] Loading and inflating from {path}...")
        with open_func(path, "rb") as f:
            data = pickle.load(f)
        
        runs = [RunResults.inflate(r, ephem=ephem) for r in data["runs"]]
        return cls(runs=runs, configs=data.get("configs"), run_ids=data.get("run_ids"))

    @property
    def satellite(self) -> Satellite:
        return self.runs[0].satellite

    @property
    def est_satellite(self) -> Optional[EstimatedSatellite]:
        return self.runs[0].est_satellite

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator[RunResults]:
        return iter(self.runs)

    def __getitem__(self, idx: Union[int, slice]) -> Union[RunResults, List[RunResults]]:
        return self.runs[idx]

    def first(self) -> RunResults:
        return self.runs[0]

    def stack_state(self) -> np.ndarray:
        return np.stack([np.asarray(r.state_hist) for r in self.runs], axis=0)

    def stack_control(self) -> np.ndarray:
        return np.stack([np.vstack(r.control_hist) for r in self.runs], axis=0)

    def stack_time(self) -> np.ndarray:
        return np.stack([np.asarray(r.time_s) for r in self.runs], axis=0)

    def map(self, attr: str) -> List[Any]:
        return [getattr(r, attr) for r in self.runs]