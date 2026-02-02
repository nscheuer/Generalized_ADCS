#!/usr/bin/env python3
"""Minimal test for bdot_on initialization."""
import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')

print("Loading imports...")
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube1_cubesat
from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR
from ADCS.controller.helpers.planner_factory import create_planner_settings
from ADCS.controller.helpers.normalized_settings import PlannerPresets
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.helpers.math_helpers import normalize, quat_mult
print("Imports done")

np.random.seed(42)
sat = create_beavercube1_cubesat(estimated=False)
print("Sat created")

orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=120, use_J2=False, fast=True)
orb.populate_environment(compute_B=True, compute_S=True)
print("Orbit done")

t_start, t_end = orb.times[5], orb.times[5] + 50
os0 = orb.get_os(t_start)

q0 = normalize(np.array([1., 0., 0., 0.]))
x0 = np.concatenate([np.zeros(3), q0]).astype(np.float64)
axis = normalize(np.array([1., 1., 0.]))

# 180 deg slew only
q_goal = normalize(np.array([0., axis[0], axis[1], axis[2]]))
goals = GoalList({t_start: Fixed_Attitude_Goal(q_goal)})

print("\n=== 180° slew - prepareForAlilqr test ===")
for bdot_mode in [0, 2]:
    config = PlannerPresets.mtq_only_normalized()
    settings = create_planner_settings(sat, config)
    settings.bdot_on = bdot_mode
    settings.dt_tp = 10
    
    controller = Plan_and_Track_PythonALILQR(est_sat=sat, planner_settings=settings, verbose=False)
    vecsPy = controller._propagate_environment(os0, t_start, t_end, 10.0, 6, goals)
    
    result = controller.planner.prepareForAlilqr(vecsPy, 10.0, t_start, t_end, x0, bdot_mode)
    Xset, Uset = np.array(result[0][0]), np.array(result[0][1])
    
    print(f"bdot={bdot_mode}: |U|={np.linalg.norm(Uset):.3e}, Uset[:,0]={Uset[:3,0]}")

print("\nDone!")
