"""
Compare PKMN (your) config vs NSSR config for ALTRO planner.
"""
import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings
from ADCS.controller.helpers.planner_subsettings import CostWeights
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize


def setup_common():
    """Common setup for both configs."""
    np.random.seed(1)
    
    real_sat = create_beavercube2_cubesat()
    
    # Initial state
    w0 = np.array([0, 0, 0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    h0 = np.array([0.0001])
    x = np.concatenate([w0, q0, h0])
    
    # Orbit setup (fake orbit for speed)
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    R = 7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V, 
                        B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12)
    
    goals = GoalList({0.22: ECI_Goal(np.array([0, 0, 1]))})
    
    return real_sat, x, os0, goals


def config_pkmn(real_sat, verbose=False):
    """Your original PKMN config (with dt_tp fix)."""
    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=3,      # Smart bdot (your choice)
        dt_tp=10,       # FIXED: was using dt_planning param
        dt_tvlqr=1,
    )
    planner_settings.verbosity = verbose
    
    # Your cost function settings
    planner_settings.cost_main.ang_cost_func_type = 0  # linear
    planner_settings.cost_main.angle = 1e6
    planner_settings.cost_main.angle_N = 1e7
    planner_settings.cost_main.ang_vel = 1e3
    planner_settings.cost_main.ang_vel_N = 1e3
    
    # Your Hessian settings
    planner_settings.cost_main.use_full_cost_hessian = False
    planner_settings.pass1.regularization.use_dynamics_hess = 0
    
    # Your iteration limits
    planner_settings.pass1.convergence.max_outer_iter = 10
    planner_settings.pass1.convergence.max_inner_iter = 60
    planner_settings.pass2.convergence.max_outer_iter = 10
    planner_settings.pass2.convergence.max_inner_iter = 25
    
    planner_settings.init_traj.bdot_gain = 500
    planner_settings.pass1.aug_lag.penalty_init = 100
    
    # Your TVLQR settings
    planner_settings.cost_tvlqr = CostWeights(
        angle=1e2,
        angle_N=1e3,
        ang_vel=1e7,
        ang_vel_N=1e8,
        control_mult=1e6,
        ang_cost_func_type=2,
    )
    
    return Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings), "PKMN (your config)"


def config_nssr(real_sat, verbose=False):
    """NSSR working config."""
    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=0,      # Skip bdot
        dt_tp=10,
        dt_tvlqr=1,
    )
    planner_settings.verbosity = verbose
    
    # NSSR Hessian settings
    planner_settings.cost_main.use_full_cost_hessian = True
    planner_settings.pass1.regularization.use_dynamics_hess = 1
    planner_settings.init_traj.bdot_gain = 500
    planner_settings.pass1.aug_lag.penalty_init = 100
    planner_settings.pass1.convergence.max_outer_iter = 8
    planner_settings.pass1.convergence.max_inner_iter = 40
    planner_settings.pass2.convergence.max_outer_iter = 5
    planner_settings.pass2.convergence.max_inner_iter = 15
    
    # NSSR cost weights
    planner_settings.cost_main = CostWeights(
        angle=1e3,
        angle_N=1e6,
        ang_vel=1e3,
        ang_vel_N=1e5,
        ang_vel_mag=0.0,
        ang_vel_mag_N=0.0,
        control_mult=1.0,
        ang_cost_func_type=2,
    )
    
    planner_settings.cost_tvlqr = CostWeights(
        angle=1e2,
        angle_N=1e3,
        ang_vel=1e6,
        ang_vel_N=1e8,
        ang_vel_mag=0.0,
        ang_vel_mag_N=0.0,
        control_mult=1.0,
        ang_cost_func_type=2,
    )
    
    return Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings), "NSSR config"


def run_test(controller, name, x, os0, goals, tf=500, verbose=False):
    """Run trajectory planning and return metrics."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    t_start = time.perf_counter()
    
    try:
        traj = controller.calculate_trajectory(
            t_start=0.22,
            duration=tf,
            x_0=x,
            os_0=os0,
            goals=goals,
            verbose=verbose
        )
        
        t_end = time.perf_counter()
        wall_time = t_end - t_start
        
        # Calculate final error
        final_state = traj.states[:, -1]
        q_final = final_state[3:7]
        w, x_q, y_q, z_q = q_final
        R = np.array([
            [1 - 2*(y_q**2 + z_q**2), 2*(x_q*y_q - z_q*w), 2*(x_q*z_q + y_q*w)],
            [2*(x_q*y_q + z_q*w), 1 - 2*(x_q**2 + z_q**2), 2*(y_q*z_q - x_q*w)],
            [2*(x_q*z_q - y_q*w), 2*(y_q*z_q + x_q*w), 1 - 2*(x_q**2 + y_q**2)]
        ])
        body_boresight = np.array([0, 1, 0])  # BC2 boresight
        eci_boresight = R @ body_boresight
        goal_eci = np.array([0, 0, 1])
        error_rad = np.arccos(np.clip(np.dot(eci_boresight, goal_eci), -1, 1))
        final_error_deg = np.rad2deg(error_rad)
        
        final_omega_deg = np.rad2deg(np.linalg.norm(final_state[:3]))
        
        print(f"\nResults for {name}:")
        print(f"  Wall time:        {wall_time:.1f}s")
        print(f"  RTF:              {wall_time/tf:.3f}x")
        print(f"  Trajectory pts:   {len(traj.times)}")
        print(f"  Final error:      {final_error_deg:.2f}°")
        print(f"  Final |ω|:        {final_omega_deg:.4f}°/s")
        
        return {
            'name': name,
            'success': True,
            'wall_time': wall_time,
            'rtf': wall_time/tf,
            'n_points': len(traj.times),
            'final_error_deg': final_error_deg,
            'final_omega_deg': final_omega_deg,
        }
        
    except Exception as e:
        t_end = time.perf_counter()
        print(f"\nFAILED: {e}")
        return {
            'name': name,
            'success': False,
            'error': str(e),
            'wall_time': t_end - t_start,
        }


def main():
    print("Setting up common environment...")
    real_sat, x, os0, goals = setup_common()
    
    tf = 500  # 500s trajectory
    verbose = False
    
    results = []
    
    # Test NSSR config first (known working)
    controller_nssr, name_nssr = config_nssr(real_sat, verbose)
    results.append(run_test(controller_nssr, name_nssr, x, os0, goals, tf, verbose))
    
    # Test PKMN config
    controller_pkmn, name_pkmn = config_pkmn(real_sat, verbose)
    results.append(run_test(controller_pkmn, name_pkmn, x, os0, goals, tf, verbose))
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Config':<20} {'Status':<10} {'Wall Time':<12} {'RTF':<10} {'Final Error':<12} {'Final |ω|':<10}")
    print("-"*70)
    
    for r in results:
        if r['success']:
            print(f"{r['name']:<20} {'OK':<10} {r['wall_time']:.1f}s{'':<6} {r['rtf']:.3f}x{'':<5} {r['final_error_deg']:.2f}°{'':<7} {r['final_omega_deg']:.4f}°/s")
        else:
            print(f"{r['name']:<20} {'FAILED':<10} {r.get('error', 'Unknown')[:40]}")
    
    print("="*70)


if __name__ == "__main__":
    main()
