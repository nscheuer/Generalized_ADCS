"""
TODO-DATA-6, TODO-DATA-7: Computational Performance Tests
=========================================================

Papers: Package Paper, Planner Paper
TODO IDs:
  - TODO-DATA-6: Profile memory and computational requirements
  - TODO-DATA-7: Measure computational scaling (vary timestep and horizon)
  - TODO-JGCD-4: Complete computational complexity analysis
  - TODO-SMALLSAT-5: Add computational requirements table (solver time, memory)
  - TODO-JGCD-6: Python vs C++ performance comparison

This module benchmarks computational performance.

Adjustable Parameters
---------------------
- N_TIMING_ITERATIONS: Number of iterations for timing
- HORIZON_LENGTHS: Planner horizon lengths to test
- TIMESTEPS: Control timesteps to test
"""

import sys
import os
import numpy as np
import pytest
import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.controller.mtq_w_rw_LP import MTQ_w_RW_LP
from ADCS.controller.mtq_w_rw_QP import MTQ_w_RW_QP
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.math_constants import MathConstants


# =============================================================================
# ADJUSTABLE PARAMETERS
# =============================================================================

# Timing parameters
N_TIMING_ITERATIONS = 100
N_WARMUP = 10

# Scaling parameters
HORIZON_LENGTHS = [10, 25, 50, 100, 200]  # Number of steps
TIMESTEPS = [0.1, 0.5, 1.0, 2.0]          # Seconds

# Memory profiling
PROFILE_MEMORY = True

# Pretty output
PRETTY_OUTPUT = True


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TimingResult:
    """Result from timing benchmark."""
    component: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_iterations: int


@dataclass
class MemoryResult:
    """Result from memory profiling."""
    component: str
    peak_mb: float
    current_mb: float


@dataclass
class ScalingResult:
    """Result from scaling analysis."""
    horizon_length: int
    timestep: float
    mean_time_ms: float
    memory_mb: float


# =============================================================================
# PRETTY OUTPUT
# =============================================================================

class PrettyOutput:
    HEADER = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    @staticmethod
    def header(text: str) -> None:
        if PRETTY_OUTPUT:
            print(f"\n{PrettyOutput.BOLD}{PrettyOutput.HEADER}{'='*70}")
            print(f"  {text}")
            print(f"{'='*70}{PrettyOutput.ENDC}\n")
    
    @staticmethod
    def timing_table(results: List[TimingResult]) -> None:
        """Print timing results as table."""
        print(f"\n{PrettyOutput.BOLD}  Timing Benchmark Results{PrettyOutput.ENDC}")
        print("  " + "─" * 70)
        
        headers = ["Component", "Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)"]
        widths = [25, 12, 12, 12, 12]
        
        header_row = "  │ " + " │ ".join(f"{h:^{w}}" for h, w in zip(headers, widths)) + " │"
        print(header_row)
        print("  ├─" + "─┼─".join("─" * w for w in widths) + "─┤")
        
        for r in results:
            cols = [
                r.component[:23],
                f"{r.mean_ms:.3f}",
                f"{r.std_ms:.3f}",
                f"{r.min_ms:.3f}",
                f"{r.max_ms:.3f}",
            ]
            row = "  │ " + " │ ".join(f"{c:^{w}}" for c, w in zip(cols, widths)) + " │"
            
            # Color code based on performance
            if PRETTY_OUTPUT:
                if r.mean_ms < 1.0:
                    print(f"{PrettyOutput.GREEN}{row}{PrettyOutput.ENDC}")
                elif r.mean_ms < 5.0:
                    print(f"{PrettyOutput.YELLOW}{row}{PrettyOutput.ENDC}")
                else:
                    print(row)
            else:
                print(row)
        
        print("  └─" + "─┴─".join("─" * w for w in widths) + "─┘")
    
    @staticmethod
    def memory_summary(results: List[MemoryResult]) -> None:
        """Print memory usage summary."""
        print(f"\n{PrettyOutput.BOLD}  Memory Usage Summary{PrettyOutput.ENDC}")
        print("  " + "─" * 45)
        
        for r in results:
            bar_len = int(r.peak_mb / 10)  # Scale for display
            bar = "█" * min(bar_len, 30)
            print(f"  {r.component:25} │ {bar} {r.peak_mb:.2f} MB")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_mtq_rw_config():
    """Create standard satellite configuration."""
    mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
    rw = RW(axis=np.array([0, 0, 1]), max_torque=0.01, J=0.001, h=0.0, h_max=0.05)
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return dict(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        actuators=mtqs + [rw],
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )


def benchmark_function(func, n_warmup: int = N_WARMUP, 
                       n_iterations: int = N_TIMING_ITERATIONS) -> TimingResult:
    """Benchmark a function's execution time."""
    # Warmup
    for _ in range(n_warmup):
        func()
    
    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    
    return TimingResult(
        component="",  # Filled by caller
        mean_ms=np.mean(times),
        std_ms=np.std(times),
        min_ms=np.min(times),
        max_ms=np.max(times),
        n_iterations=n_iterations,
    )


# =============================================================================
# TODO-DATA-6: MEMORY PROFILING TESTS
# =============================================================================

class TestMemoryProfiling:
    """
    TODO-DATA-6: Profile memory requirements.
    """

    def test_satellite_memory_usage(self):
        """Profile memory usage of satellite creation."""
        PrettyOutput.header("TODO-DATA-6: Memory Profiling")
        
        results = []
        
        # Profile satellite creation
        tracemalloc.start()
        
        config = create_mtq_rw_config()
        est_sat = EstimatedSatellite(**config)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results.append(MemoryResult(
            component="EstimatedSatellite",
            peak_mb=peak / 1024 / 1024,
            current_mb=current / 1024 / 1024,
        ))
        
        # Profile controller creation
        tracemalloc.start()
        
        controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results.append(MemoryResult(
            component="MTQ_w_RW_LP Controller",
            peak_mb=peak / 1024 / 1024,
            current_mb=current / 1024 / 1024,
        ))
        
        PrettyOutput.memory_summary(results)
        
        # Basic assertions
        for r in results:
            assert r.peak_mb < 100, f"{r.component} uses too much memory: {r.peak_mb:.2f}MB"


# =============================================================================
# TODO-DATA-7: TIMING BENCHMARKS
# =============================================================================

class TestTimingBenchmarks:
    """
    TODO-DATA-7: Benchmark computational timing.
    """

    def test_allocation_timing(self):
        """Benchmark allocation timing for LP and QP."""
        PrettyOutput.header("TODO-DATA-7: Allocation Timing Benchmarks")
        
        B = normalize(np.array([1, 1, 1])) * 3e-5
        tau_des = np.array([0.001, 0.0005, 0.0002])
        
        results = []
        
        # LP Timing
        config_lp = create_mtq_rw_config()
        est_sat_lp = EstimatedSatellite(**config_lp)
        controller_lp = MTQ_w_RW_LP(est_sat_lp, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        def lp_func():
            controller_lp.allocate_max_torque_in_direction(tau_des, B, est_sat_lp)
        
        result = benchmark_function(lp_func)
        result.component = "LP Allocation"
        results.append(result)
        
        # QP Timing
        config_qp = create_mtq_rw_config()
        est_sat_qp = EstimatedSatellite(**config_qp)
        controller_qp = MTQ_w_RW_QP(est_sat_qp, p_gain=1.0, d_gain=0.5, c_gain=0.0)
        
        def qp_func():
            controller_qp.allocate_max_torque_in_direction(tau_des, B, est_sat_qp)
        
        result = benchmark_function(qp_func)
        result.component = "QP Allocation"
        results.append(result)
        
        PrettyOutput.timing_table(results)
        
        # Both should be real-time capable
        for r in results:
            assert r.mean_ms < 10.0, f"{r.component} too slow: {r.mean_ms:.3f}ms"

    def test_dynamics_timing(self):
        """Benchmark dynamics evaluation timing."""
        PrettyOutput.header("TODO-DATA-7: Dynamics Timing")
        
        from ADCS.orbits.orbital_state import Orbital_State
        from ADCS.orbits.ephemeris import Ephemeris
        from ADCS.satellite_hardware.satellite.satellite import Satellite
        
        # Create satellite
        mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs]
        rws = [RW(axis=j, max_torque=0.01, J=0.001, h=0.0, h_max=0.05) 
               for j in MathConstants.unitvecs]
        
        sat = Satellite(
            mass=4.0,
            J_0=np.diagflat([0.1, 0.1, 0.1]),
            actuators=mtqs + rws,
        )
        
        ephem = Ephemeris()
        os = Orbital_State(
            ephem=ephem, J2000=0.22,
            R=np.array([7000, 0, 0]), V=np.array([0, 7.5, 0]),
            B=np.array([2e-5, 1e-5, 3e-5])
        )
        
        # State and control
        x = np.hstack([np.zeros(3), normalize(np.array([0, 0, 0, 1])), np.zeros(3)])
        u = np.zeros(6)  # 3 MTQ + 3 RW
        
        def dynamics_func():
            sat.dynamics_core(x, u, os)
        
        result = benchmark_function(dynamics_func, n_iterations=500)
        result.component = "Dynamics Evaluation"
        
        PrettyOutput.timing_table([result])
        
        # Should be very fast (< 1ms for real-time 1kHz control)
        assert result.mean_ms < 1.0, f"Dynamics too slow: {result.mean_ms:.3f}ms"


# =============================================================================
# TODO-SMALLSAT-5: COMPUTATIONAL REQUIREMENTS TABLE
# =============================================================================

class TestComputationalRequirementsTable:
    """
    TODO-SMALLSAT-5: Generate computational requirements table.
    """

    def test_generate_requirements_table(self):
        """Generate comprehensive computational requirements table."""
        PrettyOutput.header("TODO-SMALLSAT-5: Computational Requirements")
        
        B = normalize(np.array([1, 1, 1])) * 3e-5
        tau_des = np.array([0.001, 0.0005, 0.0002])
        
        # Test different configurations
        configs = [
            ("3 MTQ only", 3, 0),
            ("3 MTQ + 1 RW", 3, 1),
            ("3 MTQ + 3 RW", 3, 3),
        ]
        
        results = []
        
        for name, n_mtq, n_rw in configs:
            # Create config
            mtqs = [MTQ(axis=j, max_torque=0.5) for j in MathConstants.unitvecs[:n_mtq]]
            rws = [RW(axis=j, max_torque=0.01, J=0.001, h=0.0, h_max=0.05) 
                   for j in MathConstants.unitvecs[:n_rw]]
            mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
            
            config = dict(
                mass=4.0,
                J_0=np.diagflat([0.1, 0.1, 0.1]),
                actuators=mtqs + rws,
                sensors=mtms,
                boresight=np.array([0, 0, 1])
            )
            
            est_sat = EstimatedSatellite(**config)
            controller = MTQ_w_RW_LP(est_sat, p_gain=1.0, d_gain=0.5, c_gain=0.0)
            
            def alloc_func():
                controller.allocate_max_torque_in_direction(tau_des, B, est_sat)
            
            result = benchmark_function(alloc_func, n_iterations=50)
            result.component = name
            results.append(result)
        
        PrettyOutput.timing_table(results)
        
        # Summary
        print(f"\n{PrettyOutput.BOLD}  Summary for Paper Table:{PrettyOutput.ENDC}")
        print("  ┌────────────────────────┬──────────────┬───────────────┐")
        print("  │ Configuration          │ Alloc Time   │ Real-time OK? │")
        print("  ├────────────────────────┼──────────────┼───────────────┤")
        for r in results:
            ok = "✓" if r.mean_ms < 10 else "✗"
            print(f"  │ {r.component:22} │ {r.mean_ms:8.3f} ms │      {ok}        │")
        print("  └────────────────────────┴──────────────┴───────────────┘")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
