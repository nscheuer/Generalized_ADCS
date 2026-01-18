"""
Debug script for trajectory generation with 2 MTQs and 1 RW configuration.

This tests the ALTRO planner with an underactuated system where:
- 2 MTQs provide torque perpendicular to B-field
- 1 RW provides torque along a single axis

This is a more challenging configuration than the fully-actuated 3MTQ+3RW case.
"""

import sys
import os as os_pack
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Union, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os_pack.path.abspath(os_pack.path.join(__file__, "../..")))
from ADCS.CONOPS.goals import Goal, ECI_Goal, Coordinate_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings, Trajectory
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.plot_controller import plot_control, plot_rw_momentum, plot_target_tracking


def create_2mtq_1rw_satellite(
    mtq_max_torque: float = 0.5,
    rw_max_torque: float = 0.005,
    rw_J: float = 0.0014,
    rw_h0: float = 0.001,
    rw_hmax: float = 0.015,
    mtq_axes: List[np.ndarray] = None,
    rw_axis: np.ndarray = None,
    J_sat: np.ndarray = None,
) -> Tuple[Satellite, int]:
    """
    Create a satellite with 2 MTQs and 1 RW.

    Args:
        mtq_max_torque: Maximum MTQ dipole moment (Am^2)
        rw_max_torque: Maximum RW torque (Nm)
        rw_J: RW moment of inertia (kg*m^2)
        rw_h0: Initial RW angular momentum (Nms)
        rw_hmax: Maximum RW angular momentum (Nms)
        mtq_axes: List of 2 unit vectors for MTQ axes (default: X and Y)
        rw_axis: Unit vector for RW axis (default: Z)
        J_sat: Satellite inertia matrix (default: 6U cubesat-like)

    Returns:
        Tuple of (Satellite, number of RWs)
    """
    if mtq_axes is None:
        mtq_axes = [MathConstants.unitvecs[0], MathConstants.unitvecs[1]]  # X, Y
    if rw_axis is None:
        rw_axis = MathConstants.unitvecs[2]  # Z
    if J_sat is None:
        J_sat = np.diagflat([0.05, 0.08, 0.10])  # Smaller satellite

    mtqs = [MTQ(axis=ax, max_torque=mtq_max_torque) for ax in mtq_axes]
    rws = [RW(axis=rw_axis, max_torque=rw_max_torque, J=rw_J, h=rw_h0, h_max=rw_hmax)]

    acts = rws + mtqs
    rwN = len(rws)

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    sat = Satellite(
        mass=4.0,
        J_0=J_sat,
        actuators=acts,
        sensors=mtms,
        boresight=np.array([0, 0, 1])
    )

    return sat, rwN


def run_trajectory_2mtq_1rw(
    verbose: bool = False,
    tf: float = 200,
    dt: float = 1,
    real_orbit: bool = False,
    mtq_axes: List[np.ndarray] = None,
    rw_axis: np.ndarray = None,
    B_field: np.ndarray = None,
    initial_state: dict = None,
    goal_eci: np.ndarray = None,
    seed: int = 42,
) -> dict:
    """
    Run trajectory generation for 2MTQ + 1RW satellite configuration.

    Args:
        verbose: Print debug info
        tf: Final time (s)
        dt: Time step (s)
        real_orbit: Use real orbit propagation vs constant environment
        mtq_axes: MTQ axis directions
        rw_axis: RW axis direction
        B_field: Magnetic field in ECI (T), default [1e-5, 3e-5, 2e-5]
        initial_state: Dict with 'w', 'q', 'h' for initial state
        goal_eci: Target ECI pointing direction
        seed: Random seed

    Returns:
        Dict with trajectory data
    """
    np.random.seed(seed)
    t0 = 0
    N = int((tf - t0) / dt)

    # Create satellite
    sat, rwN = create_2mtq_1rw_satellite(mtq_axes=mtq_axes, rw_axis=rw_axis)

    # Initial state
    if initial_state is None:
        w0 = np.array([0.01, 0.005, -0.002])  # Small angular velocity
        q0 = normalize(np.array([1, 0.1, 0.1, 0]))  # Near identity
        h0 = np.array([0.001] * rwN)
    else:
        w0 = initial_state.get('w', np.zeros(3))
        q0 = normalize(initial_state.get('q', np.array([1, 0, 0, 0])))
        h0 = initial_state.get('h', np.array([0.001] * rwN))

    x = np.concatenate([w0, q0, h0])

    # Orbit setup
    ephem = Ephemeris()
    start_time = 0.22 - 1 * TimeConstants.sec2cent
    end_time = 0.22 + (tf - t0) * TimeConstants.sec2cent
    R = 7000 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2])
    V = np.array([7.5, 0, 0])

    if B_field is None:
        B_field = np.array([1e-5, 3e-5, 2e-5])

    if real_orbit:
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)
    else:
        os0 = Orbital_State(
            ephem=ephem,
            J2000=start_time,
            R=R, V=V,
            B=B_field,
            S=np.array([1e5+1, 0, 0]),
            rho=5e-12
        )
        dur = int((tf - t0) / dt) + 10
        orbs = [os0] * (dur + 10)
        for j in range(dur):
            orbs[j] = os0.copy()
            orbs[j].J2000 = os0.J2000 + j * dt * TimeConstants.sec2cent
        orb = Orbit(orbs)

    # Build Planner
    planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=dt)
    planner_settings.verbosity = verbose
    planner_settings.rw_control_weight = 1e4
    planner_settings.mtq_control_weight = 1e4
    planner_settings.cost_main.ang_vel = 1e4
    planner_settings.cost_second.ang_vel = 1e4
    planner_settings.cost_main.use_raw_control_cost = True
    planner_settings.pass1.aug_lag.penalty_init = 1e-3

    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)

    # Goal setup
    if goal_eci is None:
        goal_eci = normalize(np.array([1, 1, 1]))
    goals = GoalList({0.22: ECI_Goal(goal_eci)})

    print(f"\n{'='*60}")
    print("2 MTQ + 1 RW Trajectory Generation")
    print(f"{'='*60}")
    print(f"Duration: {tf} s, dt: {dt} s")
    print(f"MTQ axes: {[list(ax) for ax in (mtq_axes or [MathConstants.unitvecs[0], MathConstants.unitvecs[1]])]}")
    print(f"RW axis: {list(rw_axis or MathConstants.unitvecs[2])}")
    print(f"B-field (ECI): {B_field}")
    print(f"Initial w: {w0}")
    print(f"Initial q: {q0}")
    print(f"Goal ECI: {goal_eci}")
    print(f"{'='*60}\n")

    # Compute trajectory
    print("Computing Trajectory...")
    traj: Trajectory = controller.calculate_trajectory(
        t_start=0.22,
        duration=tf - t0,
        x_0=x,
        os_0=os0,
        goals=goals,
        verbose=verbose
    )

    controller.set_active_trajectory(traj)
    time_hist_traj = (traj.times - start_time) * TimeConstants.cent2sec
    state_hist_traj = traj.states.T
    u_hist_traj = traj.controls.T

    # Compute boresight tracking
    boresight_traj_hist = np.vstack([
        goals.to_ref(t=J2000, os0=orb.get_os(J2000))[0]
        for J2000 in traj.times
    ])

    result = {
        'times': time_hist_traj,
        'states': state_hist_traj,
        'controls': u_hist_traj,
        'boresight_goals': boresight_traj_hist,
        'satellite': sat,
        'rwN': rwN,
        'B_field': B_field,
        'goal_eci': goal_eci,
        'initial_state': {'w': w0, 'q': q0, 'h': h0},
    }

    return result


def analyze_trajectory(result: dict) -> None:
    """Analyze and print trajectory statistics."""
    times = result['times']
    states = result['states']
    controls = result['controls']
    rwN = result['rwN']

    print(f"\n{'='*60}")
    print("TRAJECTORY ANALYSIS")
    print(f"{'='*60}")
    print(f"Duration: {times[-1] - times[0]:.1f} s")
    print(f"Timesteps: {len(times)}")

    # Final state
    w_final = states[-1, 0:3]
    q_final = states[-1, 3:7]
    print(f"\nFinal angular velocity: {w_final} rad/s")
    print(f"Final angular velocity magnitude: {np.linalg.norm(w_final)*180/np.pi:.4f} deg/s")
    print(f"Final quaternion: {q_final}")

    # Control statistics
    print(f"\nControl Statistics:")
    for ch in range(controls.shape[1]):
        u_ch = controls[:, ch]
        print(f"  Channel {ch}: min={u_ch.min():.6f}, max={u_ch.max():.6f}, mean={np.abs(u_ch).mean():.6f}")

    # RW momentum if applicable
    if rwN > 0:
        h_final = states[-1, 7:7+rwN]
        print(f"\nFinal RW momentum: {h_final}")


def plot_trajectory(result: dict, show: bool = True) -> None:
    """Plot trajectory results."""
    times = result['times']
    states = result['states']
    controls = result['controls']
    boresight_goals = result['boresight_goals']
    rwN = result['rwN']

    plot_state_comparison(time=times, state_hist=states)
    plot_control(time=times, u_hist=controls)
    plot_target_tracking(
        state_hist=states,
        boresight_hist=boresight_goals,
        body_boresight=np.array([0, 0, 1])
    )

    if rwN > 0:
        plot_rw_momentum(time=times, state_hist=states)

    if show:
        plt.show()


def test_configurations():
    """Test multiple 2MTQ+1RW configurations."""

    configs = [
        {
            'name': 'XY MTQs + Z RW (aligned with B_z dominant)',
            'mtq_axes': [np.array([1, 0, 0]), np.array([0, 1, 0])],
            'rw_axis': np.array([0, 0, 1]),
            'B_field': np.array([1e-5, 1e-5, 5e-5]),  # Z-dominant B-field
        },
        {
            'name': 'XZ MTQs + Y RW',
            'mtq_axes': [np.array([1, 0, 0]), np.array([0, 0, 1])],
            'rw_axis': np.array([0, 1, 0]),
            'B_field': np.array([1e-5, 5e-5, 1e-5]),  # Y-dominant B-field
        },
        {
            'name': 'YZ MTQs + X RW',
            'mtq_axes': [np.array([0, 1, 0]), np.array([0, 0, 1])],
            'rw_axis': np.array([1, 0, 0]),
            'B_field': np.array([5e-5, 1e-5, 1e-5]),  # X-dominant B-field
        },
        {
            'name': 'Non-orthogonal MTQs',
            'mtq_axes': [
                normalize(np.array([1, 1, 0])),
                normalize(np.array([1, -1, 0]))
            ],
            'rw_axis': np.array([0, 0, 1]),
            'B_field': np.array([2e-5, 2e-5, 3e-5]),
        },
    ]

    results = []
    for config in configs:
        print(f"\n\n{'#'*70}")
        print(f"# Testing: {config['name']}")
        print(f"{'#'*70}")

        try:
            result = run_trajectory_2mtq_1rw(
                verbose=False,
                tf=100,
                dt=1,
                real_orbit=False,
                mtq_axes=config['mtq_axes'],
                rw_axis=config['rw_axis'],
                B_field=config['B_field'],
            )
            result['config_name'] = config['name']
            analyze_trajectory(result)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    """Run default 2MTQ+1RW trajectory generation with plots."""
    result = run_trajectory_2mtq_1rw(
        verbose=False,
        tf=150,
        dt=1,
        real_orbit=False,
    )

    analyze_trajectory(result)

    # Save trajectory
    save_path = os_pack.path.join(os_pack.path.dirname(__file__), "debug_2mtq_1rw_trajectory.npz")
    np.savez(
        save_path,
        times=result['times'],
        states=result['states'],
        controls=result['controls'],
        B_field=result['B_field'],
    )
    print(f"\nTrajectory saved to: {save_path}")

    plot_trajectory(result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="2MTQ + 1RW Trajectory Generation Debug")
    parser.add_argument('--test-configs', action='store_true', help='Test multiple configurations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--tf', type=float, default=150, help='Final time (s)')
    parser.add_argument('--dt', type=float, default=1, help='Time step (s)')
    parser.add_argument('--real-orbit', action='store_true', help='Use real orbit propagation')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')

    args = parser.parse_args()

    if args.test_configs:
        results = test_configurations()
        # Plot first result
        if results and not args.no_plot:
            plot_trajectory(results[0])
    else:
        result = run_trajectory_2mtq_1rw(
            verbose=args.verbose,
            tf=args.tf,
            dt=args.dt,
            real_orbit=args.real_orbit,
        )
        analyze_trajectory(result)
        if not args.no_plot:
            plot_trajectory(result)
