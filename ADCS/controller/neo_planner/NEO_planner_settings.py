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

@dataclass
class InitTrajConfig:
    initcontroller: int = 0
    
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

@dataclass
class PlannerSettings:
    est_sat: EstimatedSatellite

    # Constraints
    constraints: ConstraintConfig = field(init=False)

    # Disturbances
    disturbances: DisturbanceConfig = field(init=False)

    # Initial Guess
    init_traj: InitTrajConfig = field(default_factory=InitTrajConfig)

    # Passes
    passes: List[PassConfig] = field(default_factory=lambda: [PassConfig()])

    def __post_init__(self):
        self.disturbances = DisturbanceConfig(self.est_sat)
        self.constraints = ConstraintConfig(self.est_sat)