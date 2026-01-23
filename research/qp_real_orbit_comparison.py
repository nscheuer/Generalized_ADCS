"""
Real Orbit Controller Comparison: LP vs QP
==========================================

Uses real orbit propagation with time-varying B-field.
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Dict, Tuple
from tqdm import tqdm
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.CONOPS.goals import Goal, ECI_Goal, No_Goal
from ADCS.controller import MTQ_w_RW_LP, MTQ_w_RW_QP
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize, rot_mat

import warnings
warnings.filterwarnings('ignore')


@dataclass
class SimResult:
    name: str
    controller_name: str
    final_error_deg: float
    mean_error_deg: float
    converged: bool
    error_history: List[float]


def create_satellite_3mtq_1rw() -> Tuple[Satellite, np.ndarray]:
    """Create 3MTQ + 1RW satellite."""
    mtq_max_torque = 0.4
    mtqs = [MTQ(axis=j, max_torque=mtq_max_torque) for j in MathConstants.unitvecs]
    
    rw_max_torque = 7 * 0.001
    rw_J = 0.001
    rw_h0 = 5 * 0.001
    rw_hmax = 16.2 * 0.001
    
    rws = [RW(axis=MathConstants.unitvecs[0], max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)]
    
    acts = mtqs + rws
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    
    sat = Satellite(
        mass=1.2,
        J_0=np.diagflat([0.022, 0.022, 0.004]),
        actuators=acts,
        sensors=mtms,
        boresight=np.array([0, 0, 1]),
    )
    
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    h0 = np.array([rw_h0])
    x0 = np.concatenate([w0, q0, h0])
    
    return sat, x0


def create_real_orbit(tf: float, dt: float) -> Orbit:
    """Create real orbit with J2 propagation."""
    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + tf * TimeConstants.sec2cent
    
    R = 7000 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8.0, 0.0, 0.0])
    
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    return Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)


def compute_pointing_error(q: np.ndarray, goal_eci: np.ndarray, boresight: np.ndarray) -> float:
    """Compute pointing error in degrees."""
    R = rot_mat(q)
    boresight_eci = R @ boresight
    boresight_eci = boresight_eci / (np.linalg.norm(boresight_eci) + 1e-16)
    goal_eci = goal_eci / (np.linalg.norm(goal_eci) + 1e-16)
    c = float(np.clip(np.dot(boresight_eci, goal_eci), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def run_simulation(
    sat: Satellite,
    x0: np.ndarray,
    orb: Orbit,
    controller,
    goal: Goal,
    tf: float,
    dt: float,
    name: str,
    show_progress: bool = True,
) -> SimResult:
    """Run simulation with given controller."""
    t0 = 0
    N = int((tf - t0) / dt)
    x = x0.copy()
    
    error_history = []
    
    iterator = range(N)
    if show_progress:
        iterator = tqdm(iterator, desc=f"{name} - {controller.__class__.__name__}")
    
    for i in iterator:
        t = t0 + i * dt
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os = orb.get_os(J2000=J2000)
        
        # Get sensor readings
        sens = sat.sensor_readings(x=x, os=os)
        
        # Get control command
        u = controller.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os, goal=goal)
        
        # Store error
        goal_eci, _ = goal.to_ref(os0=os)
        err = compute_pointing_error(x[3:7], goal_eci, sat.boresight)
        error_history.append(err)
        
        # Propagate dynamics
        prev_os = os.copy()
        os_next = orb.get_os(0.22 + (t + dt) * TimeConstants.sec2cent)
        
        out = solve_ivp(
            fun=sat.dynamics_for_solver,
            t_span=(0, dt),
            y0=x,
            method="RK45",
            args=(u, prev_os, os_next),
            rtol=1e-7,
            atol=1e-7,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
    
    final_error = error_history[-1] if error_history else 180.0
    mean_error = np.mean(error_history[len(error_history)//2:])  # Mean of second half
    converged = final_error < 5.0
    
    return SimResult(
        name=name,
        controller_name=controller.__class__.__name__,
        final_error_deg=final_error,
        mean_error_deg=mean_error,
        converged=converged,
        error_history=error_history,
    )


def main():
    print("=" * 80)
    print("REAL ORBIT CONTROLLER COMPARISON: LP vs QP ALLOCATION")
    print("=" * 80)
    
    # Create satellite
    sat, x0 = create_satellite_3mtq_1rw()
    
    # Simulation parameters - full orbit
    tf = 1000  # ~17 minutes
    dt = 2
    
    print(f"\nSimulation: {tf}s with dt={dt}s")
    print("Creating orbit (this may take a moment)...")
    orb = create_real_orbit(tf, dt)
    print("Orbit created.")
    
    # Create controllers
    ctrl_lp = MTQ_w_RW_LP(
        est_sat=sat,
        p_gain=0.00005,
        d_gain=0.002,
        c_gain=0.001,
        h_target=np.array([0.005, 0.0, 0.0]),
    )
    
    ctrl_qp = MTQ_w_RW_QP(
        est_sat=sat,
        p_gain=0.00005,
        d_gain=0.002,
        c_gain=0.001,
        h_target=np.array([0.005, 0.0, 0.0]),
    )
    
    # Test configurations
    goals = [
        (ECI_Goal(np.array([0.0, 0.0, 1.0])), "Goal_Z"),
        (ECI_Goal(np.array([1.0, 0.0, 0.0])), "Goal_X"),
        (ECI_Goal(np.array([1.0, 1.0, 1.0])), "Goal_diag"),
    ]
    
    ICs = [
        (normalize(np.array([0.95, 0.2, 0.1, 0.1])), "q_small"),
        (normalize(np.array([0.7, 0.5, 0.3, 0.3])), "q_med"),
    ]
    
    all_results = []
    
    for goal, goal_name in goals:
        for q0, q_name in ICs:
            x_test = x0.copy()
            x_test[3:7] = q0
            
            config_name = f"{goal_name}|{q_name}"
            
            # Run LP
            result_lp = run_simulation(
                sat=sat, x0=x_test, orb=orb,
                controller=ctrl_lp, goal=goal,
                tf=tf, dt=dt, name=config_name,
            )
            all_results.append(result_lp)
            
            # Run QP
            result_qp = run_simulation(
                sat=sat, x0=x_test, orb=orb,
                controller=ctrl_qp, goal=goal,
                tf=tf, dt=dt, name=config_name,
            )
            all_results.append(result_qp)
            
            print(f"\n{config_name}:")
            lp_str = "✓" if result_lp.converged else "✗"
            qp_str = "✓" if result_qp.converged else "✗"
            print(f"  LP: Final={result_lp.final_error_deg:>6.2f}°, Mean={result_lp.mean_error_deg:>6.2f}° {lp_str}")
            print(f"  QP: Final={result_qp.final_error_deg:>6.2f}°, Mean={result_qp.mean_error_deg:>6.2f}° {qp_str}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    lp_results = [r for r in all_results if r.controller_name == "MTQ_w_RW_LP"]
    qp_results = [r for r in all_results if r.controller_name == "MTQ_w_RW_QP"]
    
    lp_final = [r.final_error_deg for r in lp_results]
    qp_final = [r.final_error_deg for r in qp_results]
    lp_mean = [r.mean_error_deg for r in lp_results]
    qp_mean = [r.mean_error_deg for r in qp_results]
    
    print(f"\nLP Controller:")
    print(f"  Final error: Mean={np.mean(lp_final):.2f}°, Max={np.max(lp_final):.2f}°")
    print(f"  Steady-state: Mean={np.mean(lp_mean):.2f}°")
    print(f"  Converged: {sum(r.converged for r in lp_results)}/{len(lp_results)}")
    
    print(f"\nQP Controller:")
    print(f"  Final error: Mean={np.mean(qp_final):.2f}°, Max={np.max(qp_final):.2f}°")
    print(f"  Steady-state: Mean={np.mean(qp_mean):.2f}°")
    print(f"  Converged: {sum(r.converged for r in qp_results)}/{len(qp_results)}")
    
    # Head-to-head
    lp_wins, qp_wins, ties = 0, 0, 0
    for lp_r, qp_r in zip(lp_results, qp_results):
        diff = lp_r.final_error_deg - qp_r.final_error_deg
        if abs(diff) < 0.5:
            ties += 1
        elif diff < 0:
            lp_wins += 1
        else:
            qp_wins += 1
    
    print(f"\nHead-to-head (final error):")
    print(f"  LP wins: {lp_wins}")
    print(f"  QP wins: {qp_wins}")
    print(f"  Ties:    {ties}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
