#!/usr/bin/env python
"""
Benchmark script for ADCS dynamics performance.

Measures:
- Total dynamics callback time
- Number of RHS evaluations (solve_ivp callbacks)
- Quaternion norm stability
- Per-step timing breakdown

Usage:
    python benchmark_dynamics.py [--scenario simple|heavy] [--dt 5.0] [--tf 300.0]
"""

import sys
import os
import argparse
import time
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

import ADCS
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.helpers.math_helpers import normalize


class DynamicsBenchmark:
    """Isolated dynamics performance profiler."""

    def __init__(
        self,
        satellite: ADCS.Satellite,
        os0: Orbital_State,
        os1: Optional[Orbital_State] = None,
        dt: float = 5.0,
    ):
        """Initialize benchmark environment."""
        self.satellite = satellite
        self.os0 = os0
        self.os1 = os0 if os1 is None else os1
        self.dt = dt
        self.u_current: Optional[np.ndarray] = None  # Control input for current step
        
        # Performance counters
        self.rhs_call_count = 0
        self.total_rhs_time = 0.0
        self.total_dynamics_time = 0.0
        self.rhs_times = []
        self.quat_norms = []
        self.start_time = None
        
    def _dynamics_wrapper(self, t: float, x: np.ndarray) -> np.ndarray:
        """Instrumented dynamics callback."""
        call_start = time.perf_counter()
        
        # Interpolate orbital state
        dmode = ErrorMode(add_bias=True, add_noise=True, update_bias=False, update_noise=False)
        delta_t = (self.os1.J2000 - self.os0.J2000) * TimeConstants.cent2sec
        time_frac = t / delta_t
        os = self.os0.average(self.os1, time_frac, True)
        u_current = self.u_current
        assert u_current is not None
        
        # Call dynamics
        x_dot = self.satellite.dynamics_core(x=x, u=u_current, orbital_state=os, dmode=dmode, verbose=False)
        
        call_end = time.perf_counter()
        call_time = call_end - call_start
        
        # Record metrics
        self.rhs_call_count += 1
        self.total_rhs_time += call_time
        self.rhs_times.append(call_time)
        self.quat_norms.append(np.linalg.norm(x[3:7]))
        
        return x_dot

    def propagate_step(self, x: np.ndarray, u: np.ndarray, rtol: float = 1e-7, atol: float = 1e-7) -> Tuple[np.ndarray, int]:
        """
        Propagate one step using solve_ivp with instrumented callback.
        
        Returns:
            x_next: Propagated state
            rhs_count: Number of RHS evaluations in this step
        """
        rhs_start = self.rhs_call_count
        self.u_current = u
        
        out = solve_ivp(
            fun=self._dynamics_wrapper,
            t_span=(0, self.dt),
            y0=x,
            method="RK45",
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        
        x_next = out.y[:, -1]
        x_next[3:7] = normalize(x_next[3:7])
        
        return x_next, self.rhs_call_count - rhs_start

    def run_benchmark(self, N_steps: int, rtol: float = 1e-7, atol: float = 1e-7) -> Dict:
        """
        Run benchmark over N_steps.
        
        Returns:
            Dictionary with performance metrics
        """
        # Initial state
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        x = np.array([0.001, -0.001, 0.001, q0[0], q0[1], q0[2], q0[3]])
        
        # Add RW momentum states if needed
        if self.satellite.number_RW > 0:
            x = np.concatenate([x, np.zeros(self.satellite.number_RW)])
        
        u = np.zeros(self.satellite.control_len)
        
        # Reset counters
        self.rhs_call_count = 0
        self.total_rhs_time = 0.0
        self.rhs_times = []
        self.quat_norms = []
        
        step_times = []
        rhs_per_step = []
        
        self.start_time = time.perf_counter()
        
        end_time = self.os0.J2000 + N_steps * self.dt * TimeConstants.sec2cent
        orb = Orbit(os0=self.os0, end_time=end_time, dt=self.dt, use_J2=True, fast=False)
        orbit_states = [
            orb.get_os(J2000=self.os0.J2000 + i * self.dt * TimeConstants.sec2cent)
            for i in range(N_steps + 1)
        ]

        for step in range(N_steps):
            step_start = time.perf_counter()

            self.os0 = orbit_states[step]
            self.os1 = orbit_states[step + 1]
            
            # Propagate one step
            x, rhs_count = self.propagate_step(x, u, rtol=rtol, atol=atol)
            
            step_end = time.perf_counter()
            step_times.append(step_end - step_start)
            rhs_per_step.append(rhs_count)
        
        total_elapsed = time.perf_counter() - self.start_time
        
        # Summary statistics
        quat_norms_arr = np.array(self.quat_norms)
        
        return {
            "total_steps": N_steps,
            "total_time": total_elapsed,
            "time_per_step": total_elapsed / N_steps,
            "total_rhs_calls": self.rhs_call_count,
            "rhs_calls_per_step": self.rhs_call_count / N_steps,
            "total_dynamics_time": self.total_rhs_time,
            "dynamics_fraction": self.total_rhs_time / total_elapsed if total_elapsed > 0 else 0.0,
            "mean_rhs_time": np.mean(self.rhs_times),
            "max_rhs_time": np.max(self.rhs_times),
            "min_rhs_time": np.min(self.rhs_times),
            "quat_norm_mean": np.mean(quat_norms_arr),
            "quat_norm_max_deviation": np.max(np.abs(quat_norms_arr - 1.0)),
            "rhs_times": self.rhs_times,
            "step_times": step_times,
            "rhs_per_step": rhs_per_step,
        }


def scenario_simple() -> Tuple[ADCS.Satellite, Orbital_State, Orbital_State]:
    """Simple scenario: no actuators, no disturbances."""
    sat = ADCS.Satellite(
        mass=1.0,
        J_0=np.eye(3),
        sensors=[],
        actuators=[],
        disturbances=[],
    )
    
    ephem = ADCS.Ephemeris()
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 1.0]),
    )
    os1 = Orbital_State(
        ephem=ephem,
        J2000=0.22 + 5.0 * TimeConstants.sec2cent,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 1.0]),
    )
    
    return sat, os0, os1


def scenario_heavy() -> Tuple[ADCS.Satellite, Orbital_State, Orbital_State]:
    """Heavy scenario: 3x MTQ, 1x RW, multiple sensors."""
    # MTQ actuators
    mtq_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    mtqs = [ADCS.MTQ(axis=mtq_axes[i], max_torque=0.1) for i in range(3)]
    
    # Reaction wheel
    rw = ADCS.RW(
        axis=np.array([0, 0, 1], dtype=float),
        max_torque=0.01,
        J=0.01,
        h=np.array([0.0, 0.0, 0.0]),
        h_max=np.array([10.0, 10.0, 10.0]),
    )
    
    # Sensors
    sensors = [ADCS.Gyro(axis=ax) for ax in np.eye(3)]
    
    # No disturbances in this benchmark to isolate torque computation
    disturbances = []
    
    sat = ADCS.Satellite(
        mass=10.0,
        J_0=np.diag([0.34, 0.27, 0.30]),
        sensors=sensors,
        actuators=mtqs + [rw],
        disturbances=disturbances,
    )
    
    ephem = ADCS.Ephemeris()
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 1.0]),
    )
    os1 = Orbital_State(
        ephem=ephem,
        J2000=0.22 + 5.0 * TimeConstants.sec2cent,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 1.0]),
    )
    
    return sat, os0, os1


def main():
    parser = argparse.ArgumentParser(description="ADCS dynamics benchmark")
    parser.add_argument(
        "--scenario",
        choices=["simple", "heavy"],
        default="simple",
        help="Benchmark scenario",
    )
    parser.add_argument(
        "--dt", type=float, default=5.0, help="Integration step size (s)"
    )
    parser.add_argument(
        "--tf", type=float, default=50.0, help="Total simulation time (s)"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-7, help="Integrator relative tolerance"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-7, help="Integrator absolute tolerance"
    )
    
    args = parser.parse_args()
    
    # Load scenario
    if args.scenario == "simple":
        sat, os0, os1 = scenario_simple()
    else:
        sat, os0, os1 = scenario_heavy()
    
    # Calculate number of steps
    N_steps = int(args.tf / args.dt)
    
    print(f"\n{'='*70}")
    print(f"ADCS Dynamics Benchmark: {args.scenario.upper()} scenario")
    print(f"{'='*70}")
    print(f"Scenario: {args.scenario}")
    print(f"  Total time: {args.tf} s")
    print(f"  Step size: {args.dt} s")
    print(f"  Steps: {N_steps}")
    print(f"  Actuators: {sat.control_len}")
    print(f"  Sensors: {len(sat.sensors)}")
    print(f"  Disturbances: {len(sat.disturbances)}")
    print(f"  State length: {sat.state_len}")
    print(f"Integrator settings:")
    print(f"  rtol: {args.rtol}")
    print(f"  atol: {args.atol}")
    print()
    
    # Run benchmark
    benchmark = DynamicsBenchmark(sat, os0, os1, dt=args.dt)
    results = benchmark.run_benchmark(N_steps, rtol=args.rtol, atol=args.atol)
    
    # Print results
    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Total wall time:       {results['total_time']:.3f} s")
    print(f"Time per step:         {results['time_per_step']:.4f} s")
    print(f"Total RHS calls:       {results['total_rhs_calls']}")
    print(f"RHS calls per step:    {results['rhs_calls_per_step']:.1f}")
    print(f"Total dynamics time:   {results['total_dynamics_time']:.3f} s")
    print(f"Dynamics fraction:     {results['dynamics_fraction']:.1%}")
    print()
    print("RHS callback timing:")
    print(f"  Mean:                {results['mean_rhs_time']*1e6:.2f} µs")
    print(f"  Min:                 {results['min_rhs_time']*1e6:.2f} µs")
    print(f"  Max:                 {results['max_rhs_time']*1e6:.2f} µs")
    print()
    print("Quaternion norm stability:")
    print(f"  Mean norm:           {results['quat_norm_mean']:.8f}")
    print(f"  Max deviation:       {results['quat_norm_max_deviation']:.2e}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    main()
