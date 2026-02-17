from __future__ import annotations

__all__ = ["PlannerSettings"]

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass, field

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite

from NEO_pass_settings import PassConfig
from NEO_constraint_settings import ConstraintConfig

@dataclass
class InitTrajConfig:
    initcontroller: int = 0

@dataclass
class PlannerSettings:
    est_sat: EstimatedSatellite

    # Constraints
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)

    # Initial Guess
    init_traj: InitTrajConfig = field(default_factory=InitTrajConfig)

    # Passes
    passes: List[PassConfig] = field(default_factory=lambda: [PassConfig()])
