from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MathConstants:
    unitvecs = [np.eye(3)[:,j] for j in list(range(3))]
    zeroquat  = np.array([1.0,0.0,0.0,0.0])