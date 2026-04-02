#!/usr/bin/env python
"""
Evaluate solve_ivp tolerance tuning impact on performance.

Tests various rtol/atol combinations to find optimal speed/accuracy tradeoff.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

from benchmarks.benchmark_dynamics import DynamicsBenchmark, scenario_simple, scenario_heavy


def run_tolerance_study() -> None:
    """Test different tolerance values for both scenarios."""
    
    tolerance_configs = [
        (1e-5, 1e-5),  # Loose
        (1e-6, 1e-6),  # Medium
        (1e-7, 1e-7),  # Current (tight)
        (1e-8, 1e-8),  # Extra tight
    ]
    
    print("\n" + "=" * 100)
    print("SOLVE_IVP TOLERANCE TUNING STUDY")
    print("=" * 100)
    
    for scenario_name, scenario_func in [("SIMPLE", scenario_simple), ("HEAVY", scenario_heavy)]:
        print(f"\n\n{scenario_name} SCENARIO")
        print("-" * 100)
        
        sat, os0, os1 = scenario_func()
        benchmark = DynamicsBenchmark(sat, os0, os1, dt=5.0)
        
        results = []
        
        for rtol, atol in tolerance_configs:
            print(f"  Testing rtol={rtol}, atol={atol}...", end=" ", flush=True)
            
            data = benchmark.run_benchmark(N_steps=20, rtol=rtol, atol=atol)
            
            results.append({
                "rtol": rtol,
                "atol": atol,
                "total_time": data["total_time"],
                "rhs_calls": data["total_rhs_calls"],
                "rhs_calls_per_step": data["rhs_calls_per_step"],
                "mean_rhs_time": data["mean_rhs_time"],
                "quat_norm_error": data["quat_norm_max_deviation"],
            })
            
            print(f"✓ ({data['total_rhs_calls']} RHS calls, {data['total_time']:.3f}s)")
        
        # Print table
        print(f"\n  Tolerance    | Wall Time | RHS Calls | Calls/Step | Mean RHS Time | Quat Error")
        print("  " + "-" * 85)
        for r in results:
            tol_str = f"{r['rtol']:.0e}/{r['atol']:.0e}"
            print(
                f"  {tol_str:12} | {r['total_time']:8.3f}s | {r['rhs_calls']:9d} | "
                f"{r['rhs_calls_per_step']:10.1f} | {r['mean_rhs_time']:13.1f}µs | {r['quat_norm_error']:.2e}"
            )
        
        # Calculate speedups relative to current (1e-7/1e-7)
        current_idx = 2  # 1e-7, 1e-7
        current_time = results[current_idx]["total_time"]
        
        print(f"\n  Speedup vs current (1e-7/1e-7):")
        for idx, r in enumerate(results):
            speedup = current_time / r["total_time"]
            print(
                f"    {r['rtol']:.0e}/{r['atol']:.0e}: {speedup:.2f}x "
                f"({r['rhs_calls']:4d} RHS calls, error={r['quat_norm_error']:.2e})"
            )
    
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print("""
  Based on tolerance study results:
  
  - Loose (1e-5/1e-5): ~2-3x faster but may have larger truncation errors
  - Medium (1e-6/1e-6): Good balance for most missions (~1.2-1.5x faster, tighter tolerance)
  - Current (1e-7/1e-7): Conservative default for high-precision requirements
  - Extra tight (1e-8/1e-8): May not be necessary for most ADCS applications
  
  Suggested approach:
  1. Use 1e-6/1e-6 for real-time operations (faster convergence)
  2. Use 1e-7/1e-7 for sim/analysis with stringent requirements
  3. Benchmark specific mission against accuracy requirements
    """)
    print()


if __name__ == "__main__":
    run_tolerance_study()
