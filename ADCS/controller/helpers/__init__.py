from .planner_settings import PlannerSettings
from .planner_subsettings import LineSearchConfig, AugLagConfig, RegularizationConfig, ConvergenceConfig, SolverPassConfig, CostWeights, InitTrajConfig
from .trajectory import Trajectory
from .quaternion_math import vector_alignment_error
from .debug_planner import DebugPlanner
from .build_csat import reorder_controls_cpp_to_python, reorder_gains_cpp_to_python, get_cpp_to_python_control_permutation
from .tinympc_settings import TinyMPCSettings

# Normalized settings (recommended for new code)
from .normalized_settings import (
    NormalizedPlannerConfig,
    NormalizedActuatorCosts,
    NormalizedStateCosts,
    NormalizedConstraints,
    NormalizedSettingsConverter,
    PlannerPresets,
)
from .planner_factory import create_planner_settings, estimate_conditioning

# Python ALILQR for debugging and analysis
from .python_alilqr import PythonALILQR, IterationData, OptimizationResult
from .python_alilqr_v2 import PythonALILQRv2

# Live visualization
from .live_planner_viz import LivePlannerViz, ConvergenceMonitor
from .mtq_warm_start import (
    solve_mtq_controls_body_frame,
    interpolate_trajectory_to_finer_grid,
    mtq_only_warm_start_transition,
    get_mtq_only_pass2_cost_mods,
)

__all__ = [
    # Legacy API
    "vector_alignment_error", 
    "PlannerSettings", 
    "LineSearchConfig", 
    "AugLagConfig", 
    "RegularizationConfig", 
    "ConvergenceConfig", 
    "SolverPassConfig", 
    "CostWeights", 
    "InitTrajConfig", 
    "Trajectory", 
    "DebugPlanner", 
    "reorder_controls_cpp_to_python", 
    "reorder_gains_cpp_to_python", 
    "get_cpp_to_python_control_permutation",
    # Normalized API (recommended)
    "NormalizedPlannerConfig",
    "NormalizedActuatorCosts",
    "NormalizedStateCosts",
    "NormalizedConstraints",
    "NormalizedSettingsConverter",
    "PlannerPresets",
    "create_planner_settings",
    "estimate_conditioning",
    # Python ALILQR for debugging
    "PythonALILQR",
    "IterationData", 
    "OptimizationResult",
    # Live visualization
    "LivePlannerViz",
    "ConvergenceMonitor",
]
