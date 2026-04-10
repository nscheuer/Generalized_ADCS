"""Top-level SALTRO planner configuration.

This module defines Python-side configuration objects that map to the SALTRO
planner C++ API, including disturbance modeling, constraints, pass settings,
and TVLQR gain-generation options.
"""

from __future__ import annotations

__all__ = ["PlannerSettings"]

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass, field, InitVar

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance, Prop_Disturbance

from .SALTRO_pass_settings import PassConfig
from .SALTRO_constraint_settings import ConstraintConfig

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
class InitTrajConfig:
    """Initial-trajectory generation settings.

    :param initcontroller: Initial controller selection index used by SALTRO.
    :type initcontroller: int
    """

    initcontroller: int = 2

    def to_cpp(self):
        """Convert Python settings to SALTRO C++ ``InitTrajConfig``.

        :return: C++ init-trajectory config object.
        :rtype: Any
        """
        saltro_py = _get_saltro_py()
        cpp_init = saltro_py.InitTrajConfig()
        cpp_init.initcontroller = self.initcontroller
        return cpp_init


@dataclass
class TVLQRSettings:
    """TVLQR gain-generation configuration.

    :param tvlqr_len: TVLQR horizon length in seconds.
    :type tvlqr_len: float
    :param tvlqr_overlap: Overlap between successive TVLQR windows in seconds.
    :type tvlqr_overlap: float
    """

    dt_tvlqr: float = field(init=False)
    tvlqr_len: float =  100.0
    tvlqr_overlap: float = 15.0

    def __post_init__(self):
        """Initialize derived TVLQR fields.

        ``dt_tvlqr`` is set during planner setup based on pass timestep and
        trajectory discretization, so it starts at ``0.0``.

        :return: None
        :rtype: None
        """
        self.dt_tvlqr = 0.0

    def to_cpp(self):
        """Convert Python settings to SALTRO C++ ``TVLQRSettings``.

        :return: C++ TVLQR settings object.
        :rtype: Any
        """
        saltro_py = _get_saltro_py()
        cpp_tvlqr = saltro_py.TVLQRSettings()
        cpp_tvlqr.dt_tvlqr = float(self.dt_tvlqr)
        cpp_tvlqr.tvlqr_len = float(self.tvlqr_len)
        cpp_tvlqr.tvlqr_overlap = float(self.tvlqr_overlap)
        return cpp_tvlqr
    
@dataclass
class DisturbanceConfig:
    """Disturbance-model configuration for trajectory optimization.

    The configuration controls which disturbances are included during planning
    and stores associated model coefficients and fixed torques.

    :param est_sat: Estimated satellite used to initialize disturbance terms.
    :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    """

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
        """Initialize disturbance vectors from the estimated satellite model.

        :param est_sat: Estimated satellite containing disturbance instances.
        :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        :return: None
        :rtype: None
        """
        self.res_dipole = sum([j.current_torque if isinstance(j, Dipole_Disturbance) else np.zeros(3) for j in est_sat.disturbances], start=np.zeros(3)).reshape((3,))
        self.prop_torque = sum([j.current_torque if isinstance(j, Prop_Disturbance) else np.zeros(3) for j in est_sat.disturbances], start=np.zeros(3)).reshape((3,))

        self.gendist_torq = np.zeros(3)
        self.J_est = est_sat.J_0

    def to_cpp(self):
        """Convert Python settings to SALTRO C++ ``DisturbanceConfig``.

        :return: C++ disturbance config object.
        :rtype: Any
        """
        saltro_py = _get_saltro_py()
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
    """Top-level configuration for SALTRO trajectory planning.

    This class aggregates constraints, disturbance assumptions, initial-guess
    generation, TVLQR settings, and one or more optimization passes.

    :param est_sat: Estimated satellite model used to derive constraints and
        disturbance defaults.
    :type est_sat: :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    :param init_traj: Initial trajectory generator settings.
    :type init_traj: :class:`InitTrajConfig`
    :param tvlqr: TVLQR gain-generation settings.
    :type tvlqr: :class:`TVLQRSettings`
    :param passes: Ordered list of SALTRO optimization passes.
    :type passes: list[:class:`PassConfig`]
    """

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
        """Initialize dependent constraint and disturbance configurations.

        :return: None
        :rtype: None
        """
        self.disturbances = DisturbanceConfig(self.est_sat)
        self.constraints = ConstraintConfig(self.est_sat)

    def to_cpp(self):
        """Convert Python planner settings to SALTRO C++ ``PlannerSettings``.

        The conversion copies constraints, disturbance settings, initial
        trajectory settings, TVLQR settings, and pass configurations. Passes are
        truncated to the C++ maximum supported count.

        :return: C++ planner settings object.
        :rtype: Any
        """
        saltro_py = _get_saltro_py()
        
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