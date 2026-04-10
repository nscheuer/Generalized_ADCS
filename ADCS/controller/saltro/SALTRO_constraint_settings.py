from __future__ import annotations

__all__ = ["ConstraintConfig"]

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass, field, InitVar

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite

def _get_saltro_py():
    import os
    import sys

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    build_dir = os.path.join(parent_dir, "SALTRO", "build")
    if build_dir not in sys.path:
        sys.path.append(build_dir)

    try:
        import saltro_py
    except ImportError as exc:
        raise ImportError(f"saltro_py not available (expected in {build_dir})") from exc

    return saltro_py

@dataclass
class ConstraintConfig:
    est_sat: InitVar[EstimatedSatellite]

    control_limit_scale: float = 0.75
    u_max: np.ndarray = field(init=False)
    wmax: float = 20*np.pi/180.0
    sun_limit_angle: float = 20*np.pi/180.0
    
    def __post_init__(self, est_sat):
        self.u_max = self.control_limit_scale * np.array([act.u_max for act in est_sat.actuators])

    def to_cpp(self):
        """Convert to C++ ConstraintConfig"""
        saltro_py = _get_saltro_py()
        cpp_constraints = saltro_py.ConstraintConfig()
        cpp_constraints.control_limit_scale = self.control_limit_scale
        cpp_constraints.u_max = self.u_max
        cpp_constraints.wmax = self.wmax
        cpp_constraints.sun_limit_angle = self.sun_limit_angle
        return cpp_constraints