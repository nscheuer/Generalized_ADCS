"""
Controller Comparison Test: LP vs QP Allocators
================================================

Uses the actual codebase controllers and simulation infrastructure
to compare allocation methods.
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


def create_fake_orbit(tf: float, dt: float, B_body: np.ndarray) -> Orbit:
    """Create fake orbit with constant B-field."""
    ephem = Ephemeris()
    R = 7000 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    V = np.array([8.0, 0.0, 0.0])
    
    os0 = Orbital_State(
        ephem=ephem,
        J2000=0.22 - 1 * TimeConstants.sec2cent,
        R=R, V=V,
        B=B_body,
        S=np.array([1e5 + 1.0, 0.0, 0.0]),
        rho=5e-12,
    )
    
    dur = int(tf / dt) + 10
    orbs = [os0] * (dur + 10)
    for j in range(dur):
        orbs[j] = os0.copy()
        orbs[j].J2000 = os0.J2000 + j * dt * TimeConstants.sec2cent
    return Orbit(orbs)


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
) -> SimResult:
    """Run simulation with given controller."""
    t0 = 0
    N = int((tf - t0) / dt)
    x = x0.copy()
    
    error_history = []
    
    for i in range(N):
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
    converged = final_error < 5.0
    
    return SimResult(
        name=name,
        controller_name=controller.__class__.__name__,
        final_error_deg=final_error,
        converged=converged,
        error_history=error_history,
    )


def main():
    print("=" * 80)
    print("CONTROLLER COMPARISON: LP vs QP ALLOCATION")
    print("=" * 80)
    
    # Create satellite
    sat, x0 = create_satellite_3mtq_1rw()
    
    # Simulation parameters
    tf = 500
    dt = 2
    
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
    B_fields = [
        (np.array([0.0, 30e-6, 10e-6]), "B_yz"),
        (np.array([30e-6, 0.0, 10e-6]), "B_xz"),
    ]
    
    goals = [
        (ECI_Goal(np.array([1.0, 0.0, 0.0])), "Goal_X"),
        (ECI_Goal(np.array([0.0, 1.0, 0.0])), "Goal_Y"),
    ]
    
    ICs = [
        (normalize(np.array([0.9, 0.3, 0.2, 0.1])), "q_small"),
        (normalize(np.array([0.7, 0.7, 0.0, 0.0])), "q_90deg"),
    ]
    
    all_results = []
    
    for B_body, B_name in B_fields:
        orb = create_fake_orbit(tf, dt, B_body)
        
        for goal, goal_name in goals:
            for q0, q_name in ICs:
                # Modify initial condition
                x_test = x0.copy()
                x_test[3:7] = q0
                
                config_name = f"{B_name}|{goal_name}|{q_name}"
                
                print(f"\n--- {config_name} ---")
                
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
                
                lp_str = "✓" if result_lp.converged else "✗"
                qp_str = "✓" if result_qp.converged else "✗"
                
                print(f"  LP: {result_lp.final_error_deg:>7.2f}° {lp_str}")
                print(f"  QP: {result_qp.final_error_deg:>7.2f}° {qp_str}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    lp_results = [r for r in all_results if r.controller_name == "MTQ_w_RW_LP"]
    qp_results = [r for r in all_results if r.controller_name == "MTQ_w_RW_QP"]
    
    lp_errors = [r.final_error_deg for r in lp_results]
    qp_errors = [r.final_error_deg for r in qp_results]
    
    print(f"\nLP Controller:")
    print(f"  Mean error: {np.mean(lp_errors):.2f}°")
    print(f"  Max error:  {np.max(lp_errors):.2f}°")
    print(f"  Converged:  {sum(r.converged for r in lp_results)}/{len(lp_results)}")
    
    print(f"\nQP Controller:")
    print(f"  Mean error: {np.mean(qp_errors):.2f}°")
    print(f"  Max error:  {np.max(qp_errors):.2f}°")
    print(f"  Converged:  {sum(r.converged for r in qp_results)}/{len(qp_results)}")
    
    # Head-to-head
    lp_wins = 0
    qp_wins = 0
    ties = 0
    
    for lp_r, qp_r in zip(lp_results, qp_results):
        if abs(lp_r.final_error_deg - qp_r.final_error_deg) < 0.5:
            ties += 1
        elif lp_r.final_error_deg < qp_r.final_error_deg:
            lp_wins += 1
        else:
            qp_wins += 1
    
    print(f"\nHead-to-head:")
    print(f"  LP wins: {lp_wins}")
    print(f"  QP wins: {qp_wins}")
    print(f"  Ties:    {ties}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
