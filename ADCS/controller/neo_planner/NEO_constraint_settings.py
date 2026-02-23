from __future__ import annotations

__all__ = ["ConstraintConfig"]

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass, field, InitVar

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite

@dataclass
class ConstraintConfig:
    est_sat: InitVar[EstimatedSatellite]

    control_limit_scale: float = 0.75
    u_max: np.ndarray = field(init=False)
    wmax: float = 20*np.pi/180.0
    sun_limit_angle: float = 20*np.pi/180.0
    
    def __post_init__(self, est_sat):
        self.u_max = self.control_limit_scale * np.array([act.u_max for act in est_sat.actuators])