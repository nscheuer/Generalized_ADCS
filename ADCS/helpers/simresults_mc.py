__all__ = ["MCSimulationResults"]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterator, Union

import numpy as np

from ADCS.helpers.simresults import SimulationResults
from ADCS.satellite_hardware.satellite import Satellite, EstimatedSatellite


@dataclass
class MCSimulationResults:
    runs: List[SimulationResults]
    configs: Optional[List[Dict[str, Any]]] = None
    run_ids: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.runs, list) or len(self.runs) == 0:
            raise ValueError("runs must be a non-empty list")
        for r in self.runs:
            if not isinstance(r, SimulationResults):
                raise TypeError("all runs must be SimulationResults")
        if self.configs is not None and len(self.configs) != len(self.runs):
            raise ValueError("configs length must match runs length")
        if self.run_ids is not None and len(self.run_ids) != len(self.runs):
            raise ValueError("run_ids length must match runs length")

    @property
    def satellite(self) -> Satellite:
        return self.runs[0].satellite

    @property
    def est_satellite(self) -> Optional[EstimatedSatellite]:
        return self.runs[0].est_satellite

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator[SimulationResults]:
        return iter(self.runs)

    def __getitem__(self, idx: Union[int, slice]) -> Union[SimulationResults, List[SimulationResults]]:
        return self.runs[idx]

    def first(self) -> SimulationResults:
        return self.runs[0]

    def stack_state(self) -> np.ndarray:
        return np.stack([np.asarray(r.state_hist) for r in self.runs], axis=0)

    def stack_control(self) -> np.ndarray:
        return np.stack([np.vstack(r.control_hist) for r in self.runs], axis=0)

    def stack_time(self) -> np.ndarray:
        return np.stack([np.asarray(r.time_s) for r in self.runs], axis=0)

    def map(self, attr: str) -> List[Any]:
        return [getattr(r, attr) for r in self.runs]
