__all__ = [
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
]


def __getattr__(name):
    if name in {
        "PlannerSettings",
        "LineSearchConfig",
        "AugLagConfig",
        "RegularizationConfig",
        "ConvergenceConfig",
        "SolverPassConfig",
        "CostWeights",
        "InitTrajConfig",
    }:
        from .planner_settings import PlannerSettings
        from .planner_subsettings import (
            AugLagConfig,
            ConvergenceConfig,
            CostWeights,
            InitTrajConfig,
            LineSearchConfig,
            RegularizationConfig,
            SolverPassConfig,
        )

        lookup = {
            "PlannerSettings": PlannerSettings,
            "LineSearchConfig": LineSearchConfig,
            "AugLagConfig": AugLagConfig,
            "RegularizationConfig": RegularizationConfig,
            "ConvergenceConfig": ConvergenceConfig,
            "SolverPassConfig": SolverPassConfig,
            "CostWeights": CostWeights,
            "InitTrajConfig": InitTrajConfig,
        }
        return lookup[name]

    if name == "Trajectory":
        from .trajectory import Trajectory

        return Trajectory

    if name == "vector_alignment_error":
        from .quaternion_math import vector_alignment_error

        return vector_alignment_error

    if name == "DebugPlanner":
        from .debug_planner import DebugPlanner

        return DebugPlanner

    if name in {
        "reorder_controls_cpp_to_python",
        "reorder_gains_cpp_to_python",
        "get_cpp_to_python_control_permutation",
    }:
        from .build_csat import (
            get_cpp_to_python_control_permutation,
            reorder_controls_cpp_to_python,
            reorder_gains_cpp_to_python,
        )

        lookup = {
            "reorder_controls_cpp_to_python": reorder_controls_cpp_to_python,
            "reorder_gains_cpp_to_python": reorder_gains_cpp_to_python,
            "get_cpp_to_python_control_permutation": get_cpp_to_python_control_permutation,
        }
        return lookup[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
