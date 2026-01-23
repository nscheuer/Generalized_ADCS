#!/usr/bin/env python3
"""
Trajectory Planner Comparison Runner.

This script runs comprehensive comparisons between different trajectory planners
for satellite attitude control maneuvers.

Usage:
    python run_planner_comparison.py                    # Run all comparisons
    python run_planner_comparison.py --quick            # Quick validation
    python run_planner_comparison.py --planners eigen poly  # Specific planners
    python run_planner_comparison.py --scenario RestToRest_45deg  # Specific scenario
    python run_planner_comparison.py --report results.md  # Generate report
"""
from __future__ import annotations

import sys
import os
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from testing.test_controllers.planner_comparison.base_planner import BasePlanner, PlannerConfig
from testing.test_controllers.planner_comparison.eigenaxis_trapezoidal import (
    EigenaxisTrapezoidalPlanner, TrapezoidalConfig
)
from testing.test_controllers.planner_comparison.polynomial_shaping import (
    PolynomialShapingPlanner, PolynomialConfig
)
from testing.test_controllers.planner_comparison.comparison_metrics import (
    ComparisonMetrics, print_comparison_summary
)
from testing.test_controllers.planner_comparison.test_scenarios import (
    TestScenario, ScenarioLibrary
)


def create_planners(planner_names: Optional[List[str]] = None) -> Dict[str, BasePlanner]:
    """
    Create planner instances for comparison.
    
    Args:
        planner_names: List of planner names to create. If None, creates all.
        
    Returns:
        Dictionary mapping planner names to instances
    """
    all_planners = {}
    
    # Eigenaxis + Trapezoidal (industry baseline)
    if planner_names is None or "eigen" in planner_names or "eigenaxis" in planner_names:
        config = TrapezoidalConfig(
            omega_max=0.05,
            alpha_max=0.01,
        )
        all_planners["Eigenaxis+Trapezoidal"] = EigenaxisTrapezoidalPlanner(config)
    
    # Polynomial 5th order
    if planner_names is None or "poly5" in planner_names or "polynomial" in planner_names:
        config = PolynomialConfig(
            poly_order=5,
            omega_max=0.05,
            alpha_max=0.01,
        )
        all_planners["Polynomial-5"] = PolynomialShapingPlanner(config)
    
    # Polynomial 7th order
    if planner_names is None or "poly7" in planner_names:
        config = PolynomialConfig(
            poly_order=7,
            omega_max=0.05,
            alpha_max=0.01,
        )
        all_planners["Polynomial-7"] = PolynomialShapingPlanner(config)
    
    # Direct Collocation (DIRCOL) - Note: Slow with scipy, would be faster with IPOPT
    if planner_names is not None and ("dircol" in planner_names or "collocation" in planner_names):
        try:
            from testing.test_controllers.planner_comparison.direct_collocation import (
                DirectCollocationPlanner, DirectCollocationConfig
            )
            config = DirectCollocationConfig(
                method="SLSQP",
                max_iterations=50,
                dt=10.0,  # Very coarse timestep for speed
            )
            all_planners["DIRCOL"] = DirectCollocationPlanner(config)
        except ImportError as e:
            print(f"Warning: Could not import DIRCOL planner: {e}")
    
    # Pseudospectral (Gauss-Lobatto) - Note: Slow with scipy, would be faster with specialized solvers
    if planner_names is not None and ("pseudo" in planner_names or "pseudospectral" in planner_names):
        try:
            from testing.test_controllers.planner_comparison.pseudospectral import (
                PseudospectralPlanner, PseudospectralConfig
            )
            config = PseudospectralConfig(
                n_nodes=8,  # Fewer nodes for speed
                max_iterations=50,
            )
            all_planners["Pseudospectral"] = PseudospectralPlanner(config)
        except ImportError as e:
            print(f"Warning: Could not import Pseudospectral planner: {e}")
    
    # Convex MPC
    if planner_names is None or "mpc" in planner_names or "convex" in planner_names:
        try:
            from testing.test_controllers.planner_comparison.convex_mpc import (
                ConvexMPCPlanner, ConvexMPCConfig
            )
            config = ConvexMPCConfig(
                mpc_horizon=30,
                n_linearization_iters=3,
            )
            all_planners["ConvexMPC"] = ConvexMPCPlanner(config)
        except ImportError as e:
            print(f"Warning: Could not import ConvexMPC planner: {e}")
    
    # SCP (Sequential Convex Programming) - kept for comparison
    if planner_names is None or "scp" in planner_names:
        try:
            from testing.test_controllers.planner_comparison.scp_planner import (
                SCPPlanner, SCPConfig
            )
            config = SCPConfig(
                max_scp_iterations=15,
                convergence_tol=1e-3,
                omega_max=0.05,
            )
            all_planners["SCP"] = SCPPlanner(config)
        except ImportError as e:
            print(f"Warning: Could not import SCP planner: {e}")
    
    # ALTRO wrapper (requires full ADCS infrastructure)
    if planner_names is None or "altro" in planner_names:
        try:
            from testing.test_controllers.planner_comparison.altro_wrapper import (
                ALTROWrapper, ALTROConfig
            )
            config = ALTROConfig(
                dt_tp=10.0,
                dt_tvlqr=1.0,
                pass1_max_outer_iter=10,
                pass1_max_inner_iter=50,
                pass2_max_outer_iter=8,
                pass2_max_inner_iter=30,
                use_quaternion_goal=True,
            )
            all_planners["ALTRO"] = ALTROWrapper(config)
        except ImportError as e:
            print(f"Warning: Could not import ALTRO wrapper: {e}")
            print("  ALTRO comparison will be skipped.")
    
    return all_planners


def run_comparison(
    planners: Dict[str, BasePlanner],
    scenarios: List[TestScenario],
    iterations: int = 1,
    verbose: bool = False
) -> ComparisonMetrics:
    """
    Run comparison tests for all planner/scenario combinations.
    
    Args:
        planners: Dictionary of planner instances
        scenarios: List of test scenarios
        iterations: Number of iterations per test (for timing variability)
        verbose: Whether to print progress
        
    Returns:
        ComparisonMetrics with all results
    """
    metrics = ComparisonMetrics()
    
    total_tests = len(planners) * len(scenarios) * iterations
    completed = 0
    
    print(f"\nRunning {total_tests} tests ({len(planners)} planners × {len(scenarios)} scenarios × {iterations} iter)")
    print("=" * 70)
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario.name} ({scenario.rotation_angle_deg:.1f}° maneuver)")
        print("-" * 50)
        
        for planner_name, planner in planners.items():
            for i in range(iterations):
                completed += 1
                
                if verbose:
                    print(f"  [{completed}/{total_tests}] {planner_name}...", end=" ", flush=True)
                
                try:
                    # Update planner config with scenario parameters
                    planner.config.horizon = scenario.horizon
                    planner.config.dt = scenario.dt
                    
                    # Run planner
                    if planner_name == "ALTRO":
                        # ALTRO needs special handling (full satellite infrastructure)
                        result = _run_altro_planner(planner, scenario)
                    else:
                        result = planner.solve(
                            x0=scenario.x0,
                            x_goal=scenario.x_goal,
                            J_inertia=scenario.J_inertia,
                            u_max=scenario.u_max,
                            B_field=scenario.B_field,
                        )
                    
                    # Compute metrics
                    metrics.compute_metrics(
                        planner_name=planner_name,
                        scenario_name=scenario.name,
                        planner_result=result,
                        x_goal=scenario.x_goal,
                        u_max=scenario.u_max,
                        x0=scenario.x0,
                    )
                    
                    if verbose:
                        print(f"✓ ({result.solve_time*1000:.1f}ms, err={result.final_cost:.4f})")
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
        
        # Print intermediate summary for this scenario
        if not verbose:
            _print_scenario_summary(metrics, scenario.name, planners.keys())
    
    return metrics


def _run_altro_planner(planner: BasePlanner, scenario: TestScenario) -> Any:
    """
    Run ALTRO planner with full satellite infrastructure.
    
    This creates the necessary satellite and orbital state objects.
    """
    from testing.test_controllers.planner_comparison.altro_wrapper import ALTROWrapper
    
    # Use standalone mode which creates minimal infrastructure
    return planner.solve_standalone(
        x0=scenario.x0,
        x_goal=scenario.x_goal,
        J_inertia=scenario.J_inertia,
        u_max=scenario.u_max,
        B_field=scenario.B_field,
    )


def _print_scenario_summary(metrics: ComparisonMetrics, scenario_name: str, planner_names) -> None:
    """Print summary for a single scenario."""
    print(f"  {'Planner':<25} {'Time (ms)':<12} {'Error (deg)':<12}")
    
    for r in metrics.results:
        if r.scenario_name == scenario_name:
            print(f"  {r.planner_name:<25} {r.solve_time_seconds*1000:>8.1f}    {r.final_angle_error_deg:>8.4f}")


def validate_planners(planners: Dict[str, BasePlanner]) -> None:
    """
    Run quick validation tests to ensure planners work correctly.
    """
    print("\nValidating planners...")
    print("-" * 50)
    
    # Simple 30-degree maneuver
    scenario = ScenarioLibrary.create_rest_to_rest(30.0, horizon=45.0)
    
    for name, planner in planners.items():
        print(f"  {name}...", end=" ", flush=True)
        
        try:
            planner.config.horizon = scenario.horizon
            planner.config.dt = scenario.dt
            
            if name == "ALTRO":
                result = _run_altro_planner(planner, scenario)
            else:
                result = planner.solve(
                    x0=scenario.x0,
                    x_goal=scenario.x_goal,
                    J_inertia=scenario.J_inertia,
                    u_max=scenario.u_max,
                )
            
            # Basic validation
            assert result.states is not None, "No states returned"
            assert result.controls is not None, "No controls returned"
            assert len(result.states) > 0, "Empty states"
            assert not np.any(np.isnan(result.states)), "NaN in states"
            assert not np.any(np.isnan(result.controls)), "NaN in controls"
            
            print(f"✓ (solve_time={result.solve_time*1000:.1f}ms)")
            
        except Exception as e:
            print(f"✗ FAILED: {e}")
            raise


def main():
    """Main entry point for planner comparison."""
    parser = argparse.ArgumentParser(
        description="Compare trajectory planning algorithms for satellite attitude control"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick comparison with minimal scenarios"
    )
    parser.add_argument(
        "--stress", action="store_true",
        help="Run stress test scenarios"
    )
    parser.add_argument(
        "--planners", nargs="+", default=None,
        help="Planners to compare (eigen, poly5, poly7, altro)"
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Run specific scenario by name"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=1,
        help="Number of iterations per test (for timing variability)"
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Output file for markdown report"
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Output file for JSON results"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate that planners work, don't run full comparison"
    )
    parser.add_argument(
        "--no-altro", action="store_true",
        help="Skip ALTRO planner (useful if C++ module not built)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRAJECTORY PLANNER COMPARISON")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create planners
    planner_list = args.planners
    if args.no_altro and planner_list is None:
        # Note: DIRCOL and Pseudospectral are slow with scipy, excluded by default
        # Use --planners dircol pseudo to include them explicitly
        planner_list = ["eigen", "poly5", "poly7", "mpc", "scp"]  # Fast planners only
    elif args.no_altro and planner_list is not None:
        planner_list = [p for p in planner_list if p.lower() != "altro"]
    
    planners = create_planners(planner_list)
    print(f"\nPlanners: {list(planners.keys())}")
    
    if len(planners) == 0:
        print("Error: No planners available")
        return 1
    
    # Validate planners
    try:
        validate_planners(planners)
    except Exception as e:
        print(f"\nValidation failed: {e}")
        return 1
    
    if args.validate_only:
        print("\nValidation complete!")
        return 0
    
    # Get scenarios
    if args.scenario:
        # Find specific scenario
        all_scenarios = ScenarioLibrary.get_standard_scenarios()
        scenarios = [s for s in all_scenarios if s.name == args.scenario]
        if not scenarios:
            print(f"Error: Scenario '{args.scenario}' not found")
            print(f"Available: {[s.name for s in all_scenarios]}")
            return 1
    elif args.quick:
        scenarios = ScenarioLibrary.get_quick_scenarios()
    elif args.stress:
        scenarios = ScenarioLibrary.get_stress_test_scenarios()
    else:
        scenarios = ScenarioLibrary.get_standard_scenarios()
    
    print(f"Scenarios: {[s.name for s in scenarios]}")
    
    # Run comparison
    start = time.perf_counter()
    metrics = run_comparison(
        planners=planners,
        scenarios=scenarios,
        iterations=args.iterations,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - start
    
    # Print summary
    print_comparison_summary(metrics)
    print(f"\nTotal comparison time: {elapsed:.1f}s")
    
    # Save results
    if args.json:
        metrics.save_results(args.json)
        print(f"Results saved to: {args.json}")
    
    if args.report:
        report = metrics.generate_comparison_table()
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.report}")
    
    # Auto-save to results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_json = os.path.join(results_dir, f"comparison_{timestamp}.json")
    metrics.save_results(auto_json)
    print(f"Auto-saved to: {auto_json}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
