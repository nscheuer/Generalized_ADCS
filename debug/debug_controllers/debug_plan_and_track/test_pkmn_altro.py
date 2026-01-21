"""
Test ALTRO planner on PKMN branch with NSSR-like settings.
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


def test_altro(tf: float = 500, dt: float = 1, verbose: bool = True):
    np.random.seed(1)
    
    real_sat = create_beavercube2_cubesat()
    
    # Initial state
    w0 = np.array([0, 0, 0])
    q0 = normalize(np.array([1, 0, 0, 0]))
    h0 = np.array([0.0001])
    x = np.concatenate([w0, q0, h0])
    
    # Orbit setup
    ephem = Ephemeris()
    start_time = 0.22 - 1*TimeConstants.sec2cent
    R = 7000*np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([8, 0, 0])
    
    # Fake orbit (faster)
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V, 
                        B=np.array([0, 0.1, 0]), S=np.array([1e5+1, 0, 0]), rho=5e-12)
    dur = int(tf/dt) + 10
    orbs = [os0.copy() for _ in range(dur + 10)]
    for j in range(dur):
        orbs[j].J2000 = os0.J2000 + j*dt*TimeConstants.sec2cent
    orb = Orbit(orbs)
    
    # Planner settings - matching NSSR working config
    planner_settings = PlannerSettings(
        est_sat=real_sat,
        bdot_on=0,
        dt_tp=10,  # KEY: Must be small enough so N >= 3
        dt_tvlqr=1,
    )
    planner_settings.verbosity = verbose
    
    # Cost weights from NSSR
    planner_settings.cost_main.use_full_cost_hessian = True
    planner_settings.pass1.regularization.use_dynamics_hess = 1
    planner_settings.init_traj.bdot_gain = 500
    planner_settings.cost_main.angle = 100
    planner_settings.cost_main.angle_N = 50000
    planner_settings.pass1.aug_lag.penalty_init = 100
    planner_settings.pass1.convergence.max_outer_iter = 8
    planner_settings.pass1.convergence.max_inner_iter = 40
    planner_settings.pass2.convergence.max_outer_iter = 5
    planner_settings.pass2.convergence.max_inner_iter = 15
    
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
    
    controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)
    
    goals = GoalList({0.22: ECI_Goal(np.array([0, 0, 1]))})
    
    print(f"\n========== ALTRO TRAJECTORY PLANNING (PKMN) ==========")
    print(f"Requested trajectory duration : {tf:.2f} s")
    print(f"dt_tp={planner_settings.dt_tp}, dt_tvlqr={planner_settings.dt_tvlqr}")
    
    t_start = time.perf_counter()
    
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
    
    print(f"\nTrajectory planning wall time : {wall_time:.3f} s")
    print(f"Real-time factor (RTF)        : {wall_time/tf:.3f} x")
    print(f"Trajectory points             : {len(traj.times)}")
    print("=" * 55)
    
    return traj


if __name__ == "__main__":
    # Test with 500s trajectory
    traj = test_altro(tf=500, verbose=True)
    print("\nSUCCESS!")
