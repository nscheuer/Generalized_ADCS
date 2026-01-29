"""Quick test to verify planner timing."""
import sys
import os
import time
import numpy as np

sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')
sys.path.insert(0, '/home/pmckeen/Generalized_ADCS/papers/Planner')

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings
from mc_planner_settings import create_good_planner_settings
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult

np.random.seed(42)

# Create orbit
print('Creating orbit...', flush=True)
t0 = time.time()
orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=200, use_J2=True, fast=True)
print(f'Orbit created in {time.time()-t0:.2f}s')

# Create satellite
real_sat = create_beavercube2_cubesat(estimated=False)

# Initial state - 90 degree slew
q0 = normalize(np.random.randn(4))
w0 = np.random.randn(3) * 0.5 * np.pi / 180
h0 = np.array([0.0001])

half_angle = 45 * np.pi / 180
q_rot = np.array([np.cos(half_angle), np.sin(half_angle), 0, 0])
q_goal = normalize(quat_mult(q0, q_rot))

x0 = np.concatenate([w0, q0, h0])
for i, rw in enumerate(real_sat.rw_actuators):
    rw.h = h0[i]

# Create controller with good settings
planner_settings = create_good_planner_settings(real_sat, dt_planning=1, has_rw=True)
controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)

# Plan trajectory
t_start = 0.22
os0 = orb.get_os(t_start)
goals = GoalList({t_start: Fixed_Attitude_Goal(q_goal)})

print('Planning trajectory...', flush=True)
t0 = time.time()
traj = controller.calculate_trajectory(
    t_start=t_start,
    duration=120,
    x_0=x0,
    os_0=os0,
    goals=goals,
    verbose=False  # Quiet!
)
plan_time = time.time() - t0
print(f'Planning completed in {plan_time:.2f}s')

# Check result
if traj is not None:
    # Get final state
    final_state = traj.get_state_at(traj.end_time)
    q_final = final_state[3:7]
    q_final = q_final / np.linalg.norm(q_final)
    dot = abs(np.dot(q_final, q_goal))
    angle_err = np.degrees(2 * np.arccos(min(dot, 1.0)))
    print(f'Final pointing error: {angle_err:.2f} degrees')
    print(f'Trajectory has {traj.n_steps} points, duration {traj.end_time - traj.start_time:.5f} centuries')
else:
    print('Trajectory is None!')
