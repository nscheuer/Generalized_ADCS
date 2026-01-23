"""
Comparison Metrics for Trajectory Planners.

This module provides standardized metrics for comparing trajectory planner performance.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from datetime import datetime


@dataclass
class MetricsResult:
    """Container for computed metrics from a single planner run."""
    
    # Identification
    planner_name: str
    scenario_name: str
    
    # Accuracy metrics
    final_angle_error_deg: float     # Final attitude error in degrees (full 3-DOF quaternion)
    final_pointing_error_deg: float  # Final boresight pointing error in degrees (2-DOF)
    final_omega_error: float         # Final angular velocity error (rad/s)
    max_angle_error_deg: float       # Maximum attitude error during trajectory
    rms_angle_error_deg: float       # RMS attitude error over trajectory
    
    # Performance metrics
    solve_time_seconds: float        # Wall-clock solve time
    converged: bool                  # Whether solver converged
    iterations: int                  # Number of iterations (if applicable)
    
    # Trajectory quality metrics
    control_effort: float            # Integrated |u| over trajectory
    max_control: float               # Maximum control magnitude
    smoothness: float                # Control rate (jerk) metric
    trajectory_duration: float       # Total trajectory time
    
    # Constraint metrics
    max_constraint_violation: float
    control_limit_violations: int    # Number of timesteps with limit violations
    
    # Additional info
    solver_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "planner_name": self.planner_name,
            "scenario_name": self.scenario_name,
            "final_angle_error_deg": self.final_angle_error_deg,
            "final_pointing_error_deg": self.final_pointing_error_deg,
            "final_omega_error": self.final_omega_error,
            "max_angle_error_deg": self.max_angle_error_deg,
            "rms_angle_error_deg": self.rms_angle_error_deg,
            "solve_time_seconds": self.solve_time_seconds,
            "converged": self.converged,
            "iterations": self.iterations,
            "control_effort": self.control_effort,
            "max_control": self.max_control,
            "smoothness": self.smoothness,
            "trajectory_duration": self.trajectory_duration,
            "max_constraint_violation": self.max_constraint_violation,
            "control_limit_violations": self.control_limit_violations,
            "solver_info": self.solver_info,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple runs."""
    
    planner_name: str
    n_runs: int
    
    # Statistics for each metric (mean, std, min, max)
    solve_time_mean: float
    solve_time_std: float
    solve_time_min: float
    solve_time_max: float
    
    final_error_mean: float
    final_error_std: float
    
    control_effort_mean: float
    control_effort_std: float
    
    convergence_rate: float  # Fraction of runs that converged
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "planner_name": self.planner_name,
            "n_runs": self.n_runs,
            "solve_time": {
                "mean": self.solve_time_mean,
                "std": self.solve_time_std,
                "min": self.solve_time_min,
                "max": self.solve_time_max,
            },
            "final_error": {
                "mean": self.final_error_mean,
                "std": self.final_error_std,
            },
            "control_effort": {
                "mean": self.control_effort_mean,
                "std": self.control_effort_std,
            },
            "convergence_rate": self.convergence_rate,
        }


class ComparisonMetrics:
    """
    Compute and aggregate metrics for trajectory planner comparison.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.results: List[MetricsResult] = []
    
    def compute_metrics(
        self,
        planner_name: str,
        scenario_name: str,
        planner_result: Any,  # PlannerResult
        x_goal: NDArray[np.float64],
        u_max: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]] = None,
        boresight: Optional[NDArray[np.float64]] = None,
    ) -> MetricsResult:
        """
        Compute metrics from a planner result.
        
        Args:
            planner_name: Name of the planner
            scenario_name: Name of the test scenario
            planner_result: PlannerResult object
            x_goal: Goal state for error computation
            u_max: Control limits for violation checking
            x0: Initial state (for angle error along trajectory)
            boresight: Body boresight vector for pointing error (default [0,0,1])
            
        Returns:
            MetricsResult with computed metrics
        """
        if boresight is None:
            boresight = np.array([0, 0, 1])
        
        states = planner_result.states
        controls = planner_result.controls
        times = planner_result.times
        
        N = len(times)
        dt = times[1] - times[0] if N > 1 else 1.0
        
        # Final state
        x_final = states[-1] if len(states) > 0 else np.zeros_like(x_goal)
        q_final = x_final[3:7]
        omega_final = x_final[:3]
        
        q_goal = x_goal[3:7]
        omega_goal = x_goal[:3]
        
        # Final errors
        final_angle_error = self._quaternion_angle(q_final, q_goal)
        final_pointing_error = self._pointing_error(q_final, q_goal, boresight)
        final_omega_error = np.linalg.norm(omega_final - omega_goal)
        
        # Angle error along trajectory
        angle_errors = []
        for i in range(N):
            q_i = states[i, 3:7]
            angle_errors.append(self._quaternion_angle(q_i, q_goal))
        angle_errors = np.array(angle_errors)
        
        max_angle_error = np.max(angle_errors)
        rms_angle_error = np.sqrt(np.mean(angle_errors**2))
        
        # Control metrics
        if len(controls) > 0:
            control_effort = np.sum(np.abs(controls)) * dt
            max_control = np.max(np.abs(controls))
            
            # Smoothness (control rate / jerk)
            if len(controls) > 1:
                control_diff = np.diff(controls, axis=0)
                smoothness = np.sqrt(np.mean(control_diff**2))
            else:
                smoothness = 0.0
            
            # Control limit violations
            violations = 0
            for k in range(len(controls)):
                for j in range(len(u_max)):
                    if j < controls.shape[1] and np.abs(controls[k, j]) > u_max[j] * 1.01:  # 1% tolerance
                        violations += 1
        else:
            control_effort = 0.0
            max_control = 0.0
            smoothness = 0.0
            violations = 0
        
        result = MetricsResult(
            planner_name=planner_name,
            scenario_name=scenario_name,
            final_angle_error_deg=np.degrees(final_angle_error),
            final_pointing_error_deg=np.degrees(final_pointing_error),
            final_omega_error=final_omega_error,
            max_angle_error_deg=np.degrees(max_angle_error),
            rms_angle_error_deg=np.degrees(rms_angle_error),
            solve_time_seconds=planner_result.solve_time,
            converged=planner_result.converged,
            iterations=planner_result.iterations,
            control_effort=control_effort,
            max_control=max_control,
            smoothness=smoothness,
            trajectory_duration=times[-1] - times[0] if N > 1 else 0.0,
            max_constraint_violation=planner_result.max_constraint_violation,
            control_limit_violations=violations,
            solver_info=planner_result.solver_info,
        )
        
        self.results.append(result)
        return result
    
    def aggregate_by_planner(self) -> Dict[str, AggregateMetrics]:
        """
        Compute aggregate statistics grouped by planner.
        
        Returns:
            Dictionary mapping planner name to AggregateMetrics
        """
        # Group results by planner
        by_planner: Dict[str, List[MetricsResult]] = {}
        for r in self.results:
            if r.planner_name not in by_planner:
                by_planner[r.planner_name] = []
            by_planner[r.planner_name].append(r)
        
        aggregates = {}
        for name, results in by_planner.items():
            n = len(results)
            
            solve_times = [r.solve_time_seconds for r in results]
            final_errors = [r.final_angle_error_deg for r in results]
            control_efforts = [r.control_effort for r in results]
            converged = [r.converged for r in results]
            
            aggregates[name] = AggregateMetrics(
                planner_name=name,
                n_runs=n,
                solve_time_mean=np.mean(solve_times),
                solve_time_std=np.std(solve_times),
                solve_time_min=np.min(solve_times),
                solve_time_max=np.max(solve_times),
                final_error_mean=np.mean(final_errors),
                final_error_std=np.std(final_errors),
                control_effort_mean=np.mean(control_efforts),
                control_effort_std=np.std(control_efforts),
                convergence_rate=sum(converged) / n if n > 0 else 0.0,
            )
        
        return aggregates
    
    def generate_comparison_table(self) -> str:
        """
        Generate a markdown comparison table.
        
        Returns:
            Markdown-formatted comparison table
        """
        aggregates = self.aggregate_by_planner()
        
        lines = [
            "# Trajectory Planner Comparison Results",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Statistics",
            "",
            "| Planner | Solve Time (ms) | Final Error (deg) | Control Effort | Convergence |",
            "|---------|----------------|-------------------|----------------|-------------|",
        ]
        
        for name, agg in sorted(aggregates.items()):
            lines.append(
                f"| {name} | "
                f"{agg.solve_time_mean*1000:.1f} ± {agg.solve_time_std*1000:.1f} | "
                f"{agg.final_error_mean:.2f} ± {agg.final_error_std:.2f} | "
                f"{agg.control_effort_mean:.2f} ± {agg.control_effort_std:.2f} | "
                f"{agg.convergence_rate*100:.0f}% |"
            )
        
        # Add detailed results section
        lines.extend([
            "",
            "## Detailed Results by Scenario",
            "",
        ])
        
        # Group by scenario
        by_scenario: Dict[str, List[MetricsResult]] = {}
        for r in self.results:
            if r.scenario_name not in by_scenario:
                by_scenario[r.scenario_name] = []
            by_scenario[r.scenario_name].append(r)
        
        for scenario, results in sorted(by_scenario.items()):
            lines.extend([
                f"### {scenario}",
                "",
                "| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |",
                "|---------|-----------|-----------------|---------------|--------|-----------|",
            ])
            
            for r in sorted(results, key=lambda x: x.planner_name):
                lines.append(
                    f"| {r.planner_name} | "
                    f"{r.solve_time_seconds*1000:.1f} | "
                    f"{r.final_angle_error_deg:.3f} | "
                    f"{r.final_omega_error:.4f} | "
                    f"{r.control_effort:.2f} | "
                    f"{'✓' if r.converged else '✗'} |"
                )
            
            lines.append("")
        
        return "\n".join(lines)
    
    def save_results(self, filepath: str) -> None:
        """Save results to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results],
            "aggregates": {k: v.to_dict() for k, v in self.aggregate_by_planner().items()},
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_results(self, filepath: str) -> None:
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.results = []
        for r_dict in data.get("results", []):
            self.results.append(MetricsResult(**r_dict))
    
    @staticmethod
    def _quaternion_angle(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> float:
        """Compute angle between two quaternions in radians."""
        # Normalize
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)
    
    @staticmethod
    def _pointing_error(
        q1: NDArray[np.float64],
        q2: NDArray[np.float64],
        boresight: NDArray[np.float64]
    ) -> float:
        """
        Compute pointing error between two quaternions.
        
        This measures the angle between where the boresight points in each attitude,
        ignoring roll around the boresight axis.
        
        Args:
            q1: First quaternion [qw, qx, qy, qz] (scalar-first)
            q2: Second quaternion
            boresight: Body boresight vector (what we're pointing)
            
        Returns:
            Pointing error in radians
        """
        def quat_rotate(q, v):
            """Rotate vector v by quaternion q (scalar-first: [w,x,y,z])."""
            q = q / np.linalg.norm(q)
            qw, qx, qy, qz = q
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
            ])
            return R @ v
        
        bore1 = quat_rotate(q1, boresight)
        bore2 = quat_rotate(q2, boresight)
        
        dot = np.dot(bore1, bore2)
        dot = np.clip(dot, -1.0, 1.0)
        return np.arccos(dot)


def print_comparison_summary(metrics: ComparisonMetrics) -> None:
    """Print a summary comparison to console."""
    aggregates = metrics.aggregate_by_planner()
    
    print("\n" + "="*80)
    print("TRAJECTORY PLANNER COMPARISON SUMMARY")
    print("="*80)
    
    # Header
    print(f"\n{'Planner':<25} {'Time (ms)':<15} {'Error (deg)':<15} {'Effort':<12} {'Conv %':<10}")
    print("-"*80)
    
    for name, agg in sorted(aggregates.items(), key=lambda x: x[1].solve_time_mean):
        print(
            f"{name:<25} "
            f"{agg.solve_time_mean*1000:>7.1f} ± {agg.solve_time_std*1000:<5.1f} "
            f"{agg.final_error_mean:>7.3f} ± {agg.final_error_std:<5.3f} "
            f"{agg.control_effort_mean:>8.2f}  "
            f"{agg.convergence_rate*100:>6.0f}%"
        )
    
    print("-"*80)
    print()
