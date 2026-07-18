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
from ADCS.state import EstimatedState, State

__all__ = ["RunResults", "SimulationResults"]


def _has_items(value: Any) -> bool:
    return value is not None and len(value) > 0

@dataclass
class RunResults:
    satellite: Satellite
    est_satellite: Optional[EstimatedSatellite] = None
    time_J2000: Optional[np.ndarray] = None
    time_s: Optional[np.ndarray] = None
    os_hist: Optional[List[Orbital_State]] = None
    est_os_hist: Optional[List[Orbital_State]] = None
    os_cov_hist: Optional[List[np.ndarray]] = None
    state_hist: Optional[List[State]] = None
    est_state_hist: Optional[List[EstimatedState]] = None
    state_cov_hist: Optional[List[np.ndarray]] = None
    sensor_bias: Optional[np.ndarray] = None
    est_sensor_bias: Optional[np.ndarray] = None
    actuator_bias: Optional[np.ndarray] = None
    est_actuator_bias: Optional[np.ndarray] = None
    target_hist: Optional[np.ndarray] = None
    w_target_hist: Optional[np.ndarray] = None
    boresight_hist: Optional[np.ndarray] = None
    clean_sensor_hist: Optional[np.ndarray] = None
    sensor_hist: Optional[np.ndarray] = None
    control_hist: Optional[np.ndarray] = None
    control_rpc_time_hist: Optional[np.ndarray] = None
    control_rpc_server_time_hist: Optional[np.ndarray] = None
    env_local_time_hist: Optional[np.ndarray] = None
    dynamics_time_hist: Optional[np.ndarray] = None

    def record(self, *, k: int, **kwargs):
        mapping = {
            "time_J2000": "time_J2000", "time_s": "time_s", "os": "os_hist",
            "est_os": "est_os_hist", "os_cov": "os_cov_hist", "state": "state_hist",
            "est_state": "est_state_hist", "state_cov": "state_cov_hist",
            "sensor_bias": "sensor_bias", "est_sensor_bias": "est_sensor_bias",
            "actuator_bias": "actuator_bias", "est_actuator_bias": "est_actuator_bias",
            "target": "target_hist", "w_target": "w_target_hist", "boresight": "boresight_hist",
            "clean_sensor": "clean_sensor_hist", "sensor": "sensor_hist", "control": "control_hist",
            "control_rpc_time": "control_rpc_time_hist", "control_rpc_server_time": "control_rpc_server_time_hist",
            "env_local_time": "env_local_time_hist", "dynamics_time": "dynamics_time_hist",
        }
        for key, val in kwargs.items():
            if val is not None and key in mapping:
                attr = mapping[key]
                if getattr(self, attr) is None:
                    setattr(self, attr, [])
                if key in ["state", "est_state"]:
                    getattr(self, attr).append(val.copy())
                elif key in ["target", "w_target", "boresight", "clean_sensor", "sensor", "control"]:
                    getattr(self, attr).append(np.asarray(val).copy())
                else:
                    getattr(self, attr).append(val)

    def flatten(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        if _has_items(data.get("os_hist")):
            data["os_hist"] = [os.to_dict() if hasattr(os, "to_dict") else os for os in data["os_hist"]]
        if _has_items(data.get("est_os_hist")):
            data["est_os_hist"] = [os.to_dict() if hasattr(os, "to_dict") else os for os in data["est_os_hist"]]
        if _has_items(data.get("state_hist")):
            data["state_hist"] = [state.to_dict() for state in data["state_hist"]]
        if _has_items(data.get("est_state_hist")):
            data["est_state_hist"] = [state.to_dict() for state in data["est_state_hist"]]
        return data

    @classmethod
    def inflate(cls, data: Dict[str, Any], ephem: Any = None) -> RunResults:
        if _has_items(data.get("state_hist")):
            data["state_hist"] = [
                item if isinstance(item, State) else (
                    State.from_dict(item) if isinstance(item, dict) else State.from_array(item)
                )
                for item in data["state_hist"]
            ]
        if _has_items(data.get("est_state_hist")):
            est_sat = data.get("est_satellite")
            covariances = data.get("state_cov_hist")
            if covariances is None:
                covariances = []
            converted = []
            for index, item in enumerate(data["est_state_hist"]):
                if isinstance(item, EstimatedState):
                    converted.append(item)
                elif isinstance(item, dict):
                    converted.append(EstimatedState.from_dict(item))
                else:
                    cov = covariances[index] if index < len(covariances) else None
                    converted.append(EstimatedState.from_estimator_array(
                        item,
                        n_rw=getattr(est_sat, "number_RW", 0),
                        n_act_bias=getattr(est_sat, "act_bias_len", 0),
                        n_sens_bias=getattr(est_sat, "att_sens_bias_len", 0),
                        n_dist_param=getattr(est_sat, "dist_param_len", 0),
                        cov=cov,
                    ))
            data["est_state_hist"] = converted
        if ephem is not None:
            if _has_items(data.get("os_hist")):
                data["os_hist"] = [Orbital_State.from_dict(d, ephem=ephem) for d in data["os_hist"]]
            if _has_items(data.get("est_os_hist")):
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
            "schema_version": 2,
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
        return np.stack([State.stack(r.state_hist) for r in self.runs], axis=0)

    def stack_estimated_state(self) -> np.ndarray:
        return np.stack(
            [np.vstack([state.as_estimator_array() for state in r.est_state_hist]) for r in self.runs],
            axis=0,
        )

    def stack_control(self) -> np.ndarray:
        return np.stack([np.vstack(r.control_hist) for r in self.runs], axis=0)

    def stack_time(self) -> np.ndarray:
        return np.stack([np.asarray(r.time_s) for r in self.runs], axis=0)

    def map(self, attr: str) -> List[Any]:
        return [getattr(r, attr) for r in self.runs]
