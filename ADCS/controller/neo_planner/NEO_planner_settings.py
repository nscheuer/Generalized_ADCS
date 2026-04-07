from __future__ import annotations

__all__ = ["PlannerSettings"]

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass, field, InitVar

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance, Prop_Disturbance

from .NEO_pass_settings import PassConfig
from .NEO_constraint_settings import ConstraintConfig

try:
    import sys
    import os
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    build_dir = os.path.join(parent_dir, "SALTRO", "build")
    if build_dir not in sys.path:
        sys.path.append(build_dir)
    import saltro_py
except ImportError:
    saltro_py = None

@dataclass
class InitTrajConfig:
    initcontroller: int = 2

    def to_cpp(self):
        """Convert to C++ InitTrajConfig"""
        if saltro_py is None:
            raise ImportError("saltro_py not available")
        cpp_init = saltro_py.InitTrajConfig()
        cpp_init.initcontroller = self.initcontroller
        return cpp_init


@dataclass
class TVLQRSettings:
    dt_tvlqr: float = 1.0
    tvlqr_len: float = 60.0
    tvlqr_overlap: float = 15.0

    def to_cpp(self):
        """Convert to C++ TVLQRSettings"""
        if saltro_py is None:
            raise ImportError("saltro_py not available")
        cpp_tvlqr = saltro_py.TVLQRSettings()
        cpp_tvlqr.dt_tvlqr = float(self.dt_tvlqr)
        cpp_tvlqr.tvlqr_len = float(self.tvlqr_len)
        cpp_tvlqr.tvlqr_overlap = float(self.tvlqr_overlap)
        return cpp_tvlqr
    
@dataclass
class DisturbanceConfig:
    est_sat: InitVar[EstimatedSatellite]

    plan_for_aero: int = 0
    plan_for_prop: int = 0
    plan_for_srp: int = 0
    plan_for_gg: int = 0
    plan_for_gendist: int = 0
    plan_for_resdipole: int = 0

    srp_coeff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    drag_coeff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    coeff_N: int = 0

    res_dipole: np.ndarray = field(init=False)
    prop_torque: np.ndarray = field(init=False)
    gendist_torq: np.ndarray = field(init=False)
    J_est: np.ndarray = field(init=False)

    def __post_init__(self, est_sat):
        self.res_dipole = sum([j.current_torque if isinstance(j, Dipole_Disturbance) else np.zeros(3) for j in est_sat.disturbances], start=np.zeros(3)).reshape((3,))
        self.prop_torque = sum([j.current_torque if isinstance(j, Prop_Disturbance) else np.zeros(3) for j in est_sat.disturbances], start=np.zeros(3)).reshape((3,))

        self.gendist_torq = np.zeros(3)
        self.J_est = est_sat.J_0

    def to_cpp(self):
        """Convert to C++ DisturbanceConfig"""
        if saltro_py is None:
            raise ImportError("saltro_py not available")
        cpp_dist = saltro_py.DisturbanceConfig()
        cpp_dist.plan_for_aero = bool(self.plan_for_aero)
        cpp_dist.plan_for_prop = bool(self.plan_for_prop)
        cpp_dist.plan_for_srp = bool(self.plan_for_srp)
        cpp_dist.plan_for_gg = bool(self.plan_for_gg)
        cpp_dist.plan_for_gendist = bool(self.plan_for_gendist)
        cpp_dist.plan_for_resdipole = bool(self.plan_for_resdipole)
        cpp_dist.srp_coeff = self.srp_coeff
        cpp_dist.drag_coeff = self.drag_coeff
        cpp_dist.coeff_N = self.coeff_N
        cpp_dist.res_dipole = self.res_dipole
        cpp_dist.prop_torque = self.prop_torque
        cpp_dist.gendist_torque = self.gendist_torq  # Note: Python uses gendist_torq, C++ uses gendist_torque
        cpp_dist.J_est = self.J_est
        return cpp_dist

@dataclass
class PlannerSettings:
    est_sat: EstimatedSatellite

    # Constraints
    constraints: ConstraintConfig = field(init=False)

    # Disturbances
    disturbances: DisturbanceConfig = field(init=False)

    # Initial Guess
    init_traj: InitTrajConfig = field(default_factory=InitTrajConfig)

    # TVLQR gain-generation settings
    tvlqr: TVLQRSettings = field(default_factory=TVLQRSettings)

    # Passes
    passes: List[PassConfig] = field(default_factory=lambda: [PassConfig()])

    def __post_init__(self):
        self.disturbances = DisturbanceConfig(self.est_sat)
        self.constraints = ConstraintConfig(self.est_sat)

    def to_cpp(self):
        """Convert Python PlannerSettings to C++ PlannerSettings"""
        if saltro_py is None:
            raise ImportError("saltro_py not available")
        
        cpp_settings = saltro_py.PlannerSettings()
        
        # Convert constraints
        cpp_settings.constraints = self.constraints.to_cpp()
        
        # Convert disturbances
        cpp_settings.disturbances = self.disturbances.to_cpp()
        
        # Convert init trajectory
        cpp_settings.init_traj = self.init_traj.to_cpp()

        # TVLQR gain-generation configuration
        cpp_settings.tvlqr = self.tvlqr.to_cpp()
        
        # Set number of passes
        cpp_settings.num_passes = len(self.passes)
        
        # Convert each pass in place (limited by MAX_OUTER_PASSES in C++)
        for i, pass_cfg in enumerate(self.passes):
            if i >= 2:  # MAX_OUTER_PASSES = 2
                break

            cpp_pass = cpp_settings.passes[i]
            cpp_pass.cost = pass_cfg.cost.to_cpp()
            cpp_pass.auglag = pass_cfg.aug_lag.to_cpp()
            cpp_pass.ilqr = pass_cfg.ilqr.to_cpp()
            cpp_pass.reg = pass_cfg.reg.to_cpp()
            cpp_pass.linesearch = pass_cfg.linesearch.to_cpp()
            cpp_pass.dt = pass_cfg.dt
        
        return cpp_settings