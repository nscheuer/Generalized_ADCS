from __future__ import annotations

__all__ = ["ConstraintConfig"]

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass, field

@dataclass
class ConstraintConfig:
    control_limit_scale: float = 0.75
    
    wmax: float = 20*np.pi/180.0
    sun_limit_angle: float = 20*np.pi/180.0