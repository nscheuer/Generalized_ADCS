"""Constraint configuration for SALTRO planner settings.

This module defines Python-side configuration containers that are converted
into SALTRO C++ constraint structs.
"""

from __future__ import annotations

__all__ = ["ConstraintConfig"]

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass, field, InitVar

from ADCS.satellite_hardware.satellite.satellite import Satellite

def _get_saltro_py():
    """Import and return the ``saltro_py`` binding module.

    The loader appends ``SALTRO/build`` to ``sys.path`` if needed.

    :return: Imported ``saltro_py`` module.
    :rtype: module
    :raises ImportError: If the SALTRO Python extension is not available.
    """
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
    r"""Constraint settings for SALTRO optimization.

    The class stores control and state-like bounds enforced by the optimizer,
    including actuator limits, maximum angular velocity, and sun-angle limits.

    :param est_sat: Estimated satellite model used to derive actuator limits.
    :type est_sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
    :param control_limit_scale: Scale factor applied to actuator ``u_max``.
    :type control_limit_scale: float
    :param wmax: Maximum angular-velocity magnitude (rad/s).
    :type wmax: float
    :param sun_limit_angle: Minimum sun-avoidance angle (rad).
    :type sun_limit_angle: float
    """

    est_sat: InitVar[Satellite]

    control_limit_scale: float = 0.75
    rw_momentum_limit_scale: float = 1.0
    u_max: np.ndarray = field(init=False)
    wmax: float = 20*np.pi/180.0
    sun_limit_angle: float = 20*np.pi/180.0
    
    def __post_init__(self, est_sat):
        """Initialize derived actuator limits from the estimated satellite.

        :param est_sat: Estimated satellite model containing actuators.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`
        :return: None
        :rtype: None
        """
        self.u_max = self.control_limit_scale * np.array([act.u_max for act in est_sat.actuators])

    def to_cpp(self):
        """Convert Python constraints to the SALTRO C++ ``ConstraintConfig``.

        :return: C++ constraints object.
        :rtype: Any
        """
        saltro_py = _get_saltro_py()
        cpp_constraints = saltro_py.ConstraintConfig()
        cpp_constraints.control_limit_scale = self.control_limit_scale
        cpp_constraints.rw_momentum_limit_scale = self.rw_momentum_limit_scale
        cpp_constraints.u_max = self.u_max
        cpp_constraints.wmax = self.wmax
        cpp_constraints.sun_limit_angle = self.sun_limit_angle
        return cpp_constraints
