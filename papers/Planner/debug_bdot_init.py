#!/usr/bin/env python3
"""Debug bdot_on initialization for different slew angles."""
import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')

from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube1_cubesat
from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
from ADCS.controller.helpers.planner_factory import create_planner_settings
from ADCS.controller.helpers.normalized_settings import PlannerPresets
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.helpers.math_helpers import normalize, quat_mult

np.random.seed(42)
sat = create_beavercube1_cubesat(estimated=False)
orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=150, use_J2=True, fast=True)
orb.populate_environment(compute_B=True, compute_S=True)

t_start = orb.times[10]
t_end = t_start + 100
os0 = orb.get_os(t_start)

q0 = normalize(np.array([1., 0., 0., 0.]))
x0 = np.concatenate([np.zeros(3), q0]).astype(np.float64)

axis = normalize(np.array([1., 1., 0.]))

print("=== Testing prepareForAlilqr initial trajectory ===\n")

for slew_deg in [90, 179, 180]:
    if slew_deg == 180:
        q_goal = normalize(np.array([0., axis[0], axis[1], axis[2]]))
    else:
        half = np.radians(slew_deg) / 2
        q_goal = normalize(np.array([np.cos(half), axis[0]*np.sin(half), axis[1]*np.sin(half), axis[2]*np.sin(half)]))
    
    goals = GoalList({t_start: Fixed_Attitude_Goal(q_goal)})
    print(f"--- Slew: {slew_deg}° ---")
    
    for bdot_mode in [0, 1, 2, 3]:
        config = PlannerPresets.mtq_only_normalized()
        settings = create_planner_settings(sat, config)
        settings.bdot_on = bdot_mode
        settings.dt_tp = 10
        
        controller = Plan_and_Track_PythonALILQR(est_sat=sat, planner_settings=settings, verbose=False)
        
        dt = 10.0
        N = 11
        vecsPy = controller._propagate_environment(os0, t_start, t_end, dt, N, goals)
        
        result = controller.planner.prepareForAlilqr(vecsPy, dt, t_start, t_end, x0, bdot_mode)
        Xset, Uset = np.array(result[0][0]), np.array(result[0][1])
        
        ctrl_norm = np.linalg.norm(Uset)
        ctrl_max = np.max(np.abs(Uset)) if Uset.size > 0 else 0
        
        print(f"  bdot={bdot_mode}: |U|={ctrl_norm:.3e}, max={ctrl_max:.3e}")
    print()

print("Done!")
