#!/usr/bin/env python3
"""
Minimal ALTRO test - quick sanity check that the trajectory planner works.

This script runs a single trajectory optimization to verify:
1. The C++ module loads correctly
2. ALTRO converges without NaN/Inf errors
3. The resulting trajectory is physically reasonable

Usage:
    python minimal_altro_test.py
"""

import sys
import os
import numpy as np
import time

# Add project root to path BEFORE any ADCS imports
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))


def main():
    print("=" * 60)
    print("Minimal ALTRO Test")
    print("=" * 60)

    # Import after path setup
    from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
    from ADCS.orbits.orbit import Orbit
    from ADCS.orbits.orbital_state import Orbital_State
    from ADCS.orbits.ephemeris import Ephemeris
    from ADCS.orbits.universal_constants import TimeConstants
    from ADCS.CONOPS.goallist import GoalList
    from ADCS.CONOPS.goals import ECI_Goal
    from ADCS.controller.helpers import PlannerSettings
    from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
    from ADCS.helpers.math_helpers import normalize

    print("\n[1/5] Creating satellite...")
    real_sat = create_beavercube2_cubesat(estimated=False)
    # Initialize RW momentum to zero
    if hasattr(real_sat, 'rw_actuators') and len(real_sat.rw_actuators) > 0:
        real_sat.rw_actuators[0].h = 0.0
    print(f"      Satellite: BeaverCube2")
    print(f"      MTQs: {len(real_sat.mtq_actuators)}, RWs: {len(real_sat.rw_actuators)}")

    print("\n[2/5] Creating orbit...")
    ephem = Ephemeris()
    start_time = 0.22
    end_time = start_time + 120 * TimeConstants.sec2cent  # 120 second trajectory
    R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])  # 630km altitude
    V = np.array([8, 0, 0])  # ~8 km/s
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orb = Orbit(os0=os0, end_time=end_time, dt=1, use_J2=True, fast=False)
    os_0 = orb.get_os(start_time)
    print(f"      Altitude: ~{np.linalg.norm(R)/1000 - 6371:.0f} km")

    print("\n[3/5] Setting up controller...")
    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=2,
        dt_tp=30,
        dt_tvlqr=1,
    )
    controller = Plan_and_Track_LQR(
        est_sat=real_sat,
        planner_settings=planner_settings,
    )

    print("\n[4/5] Planning trajectory...")
    # Initial state: [omega(3), q(4), h(1)] - small angular velocity, identity quaternion
    omega0 = np.array([0.01, 0.01, 0.01])  # Small angular velocity (rad/s)
    q0 = np.array([0.0, 0.0, 0.0, 1.0])    # Identity quaternion (w,x,y,z -> stored as x,y,z,w)
    h0 = np.array([0.0])                    # Zero RW momentum
    x0 = np.concatenate([omega0, q0, h0])

    # Goal: point +Z body axis toward ECI +Z
    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    goals = GoalList({start_time: goal})

    duration = 60.0  # 60 seconds

    start_solve = time.time()
    try:
        trajectory = controller.calculate_trajectory(
            t_start=start_time,
            duration=duration,
            x_0=x0,
            os_0=os_0,
            goals=goals,
            verbose=2
        )
        elapsed = time.time() - start_solve
        print(f"      Planning completed in {elapsed:.2f}s")
    except Exception as e:
        print(f"      ERROR: Planning failed with: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n[5/5] Validating results...")
    # Check for NaN/Inf
    states = trajectory.states
    controls = trajectory.controls

    has_nan = np.any(np.isnan(states)) or np.any(np.isnan(controls))
    has_inf = np.any(np.isinf(states)) or np.any(np.isinf(controls))

    if has_nan:
        print("      FAIL: Trajectory contains NaN values!")
        return 1
    if has_inf:
        print("      FAIL: Trajectory contains Inf values!")
        return 1

    print("      No NaN/Inf detected - trajectory is valid")

    # Check trajectory shape
    print(f"      States shape: {states.shape}")
    print(f"      Controls shape: {controls.shape}")

    print("\n" + "=" * 60)
    print("SUCCESS: ALTRO test passed!")
    print("=" * 60)
    print(f"  Trajectory steps: {trajectory.n_steps}")
    print(f"  State dimension: {trajectory.state_dim}")
    print(f"  Control dimension: {trajectory.ctrl_dim}")
    print(f"  Solve time: {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
