from dataclasses import dataclass
from typing import Optional
import numpy as np

from ADCS.orbits.orbit import Orbit
from ADCS.CONOPS.goals import Goal


@dataclass
class MCConfig:
    w: Optional[np.ndarray] = None         
    q: Optional[np.ndarray] = None          
    h: Optional[np.ndarray] = None          

    orbit: Optional[Orbit] = None          

    goal: Optional[Goal] = None

    dt: Optional[float] = None
    tf: Optional[float] = None