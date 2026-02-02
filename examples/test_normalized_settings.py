"""
Test the normalized planner settings.

Compares:
1. Legacy manual settings (dissertation-style)
2. Normalized auto-scaled settings
3. Convergence behavior and trajectory quality
"""

import sys
import os
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import (
    PlannerSettings,
    create_planner_settings,
    estimate_conditioning,
    NormalizedPlannerConfig,
    NormalizedActuatorCosts,
    NormalizedStateCosts,
    NormalizedConstraints,
    PlannerPresets,
)
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize, quat_mult


def create_legacy_settings(real_sat):
    """Create settings using legacy manual approach (dissertation-style with fix)."""
    planner_settings = PlannerSettings(est_sat=real_sat, bdot_on=0)
    
    # Manually tuned weights (had to figure out 2e7 for BeaverCube2's small RW)
    planner_settings.mtq_control_weight = 1e3
    planner_settings.rw_control_weight = 2e7  # Scaled for 1.73 mNm RW
    planner_settings.rw_AM_weight = 1e4
    planner_settings.wmax = 20 * math.pi / 180.0
    planner_settings.verbosity = False
    
    return planner_settings


def create_normalized_settings(real_sat, verbose=False):
    """Create settings using normalized auto-scaling approach."""
    config = NormalizedPlannerConfig(
        actuator_costs=NormalizedActuatorCosts(
            mtq_cost=1.0,       # Baseline
            rw_torque_cost=5.0, # RW slightly more expensive
            rw_momentum_cost=10.0,
        ),
        state_costs=NormalizedStateCosts(
            angle_cost=100.0,
            angle_terminal_cost=1000.0,
            ang_vel_cost=100.0,
            ang_vel_terminal_cost=1000.0,
        ),
        constraints=NormalizedConstraints(
            max_angular_velocity_deg_s=20.0,
            control_margin=0.25,
        ),
    )
    
    return create_planner_settings(real_sat, config=config, bdot_on=0, verbose=verbose)


def create_preset_settings(real_sat, preset_name, verbose=False):
    """Create settings using a preset."""
    return create_planner_settings(real_sat, preset=preset_name, bdot_on=0, verbose=verbose)


def run_trajectory_test(planner_settings, real_sat, orb, x0, q_goal, tf=120, label=""):
    """Run a single trajectory optimization and closed-loop simulation."""
    
    controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)
    
    goals = GoalList({0.22: Fixed_Attitude_Goal(q_goal)})
    os0 = orb.get_os(0.22)
    
    # Time trajectory generation
    t_start = time.time()
    try:
        traj = controller.calculate_trajectory(
            t_start=0.22, duration=tf, x_0=x0, os_0=os0, goals=goals, verbose=False
        )
        controller.set_active_trajectory(traj)
        traj_time = time.time() - t_start
        traj_valid = True
    except Exception as e:
        print(f"  {label}: Trajectory generation FAILED: {e}")
        return None
    
    # Analyze trajectory controls
    rw_ctrl = traj.controls[3, :]
    sign_changes = np.sum(np.diff(np.sign(rw_ctrl)) != 0)
    sign_change_rate = sign_changes / max(len(rw_ctrl) - 1, 1) * 100
    
    # Run closed-loop simulation
    N = int(tf)
    dt = 1
    time_hist = np.zeros(N)
    att_error_hist = np.zeros(N)
    
    for i, rw in enumerate(real_sat.rw_actuators):
        rw.h = x0[7 + i]
    
    x = x0.copy()
    t = 0
    sec2cent = TimeConstants.sec2cent
    
    for i in range(N):
        J2000 = 0.22 + t * sec2cent
        os_state = orb.get_os(J2000=J2000)
        sens = real_sat.sensor_readings(x=x, os=os_state)
        u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state)
        
        time_hist[i] = t
        q_curr = x[3:7]
        att_error_hist[i] = 2 * np.arccos(np.clip(np.abs(np.dot(q_curr, q_goal)), -1, 1)) * 180 / np.pi
        
        t += dt
        os_next = orb.get_os(0.22 + t * sec2cent)
        out = solve_ivp(
            real_sat.dynamics_for_solver, (0, dt), x, method="RK45",
            args=(u, os_state, os_next), rtol=1e-7, atol=1e-7
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
    
    return {
        'label': label,
        'traj_time': traj_time,
        'final_error': att_error_hist[-1],
        'min_error': np.min(att_error_hist),
        'sign_change_rate': sign_change_rate,
        'time_hist': time_hist,
        'att_error_hist': att_error_hist,
        'controls': traj.controls,
    }


def main():
    print("=" * 70)
    print("NORMALIZED PLANNER SETTINGS TEST")
    print("=" * 70)
    
    # Setup
    np.random.seed(42)
    tf = 180
    
    real_sat = create_beavercube2_cubesat(estimated=False)
    orb = create_random_circular_orbit(radius_km=7000.0, dt=1, tf=tf, use_J2=True, fast=True)
    orb.populate_environment(compute_B=True, compute_S=True)
    
    # Initial conditions - 90 degree slew
    rng = np.random.default_rng(seed=1000)
    q0 = normalize(rng.standard_normal(4))
    w0 = normalize(rng.standard_normal(3)) * (0.5 * np.pi / 180.0)
    h0 = rng.uniform(-0.0001, 0.0001, size=1)
    
    rand_angle = rng.uniform(0, 2 * np.pi)
    axis_body = np.array([np.cos(rand_angle), 0, np.sin(rand_angle)])
    half_angle = 45 * np.pi / 180
    q_rot = np.array([np.cos(half_angle), axis_body[0]*np.sin(half_angle), 
                      axis_body[1]*np.sin(half_angle), axis_body[2]*np.sin(half_angle)])
    q_goal = normalize(quat_mult(q0, q_rot))
    
    x0 = np.concatenate([w0, q0, h0])
    for i, rw in enumerate(real_sat.rw_actuators):
        rw.h = h0[i]
    
    print(f"\nTest setup:")
    print(f"  Initial attitude error: 90 deg")
    print(f"  Simulation time: {tf} s")
    print(f"  Satellite: BeaverCube2 (3 MTQ + 1 RW)")
    print(f"  MTQ max: {real_sat.mtq_actuators[0].u_max} Am²")
    print(f"  RW max: {real_sat.rw_actuators[0].u_max * 1000:.2f} mNm")
    
    # === Compare conditioning ===
    print("\n" + "=" * 70)
    print("CONDITIONING COMPARISON")
    print("=" * 70)
    
    # Legacy settings
    legacy_settings = create_legacy_settings(real_sat)
    print("\n[Legacy Settings - Manual Tuning]")
    print(f"  mtq_control_weight: {legacy_settings.mtq_control_weight:.0e}")
    print(f"  rw_control_weight: {legacy_settings.rw_control_weight:.0e}")
    
    # Compute effective Quu diagonal entries
    mtq_umax = real_sat.mtq_actuators[0].u_max * 0.75
    rw_umax = real_sat.rw_actuators[0].u_max * 0.75
    legacy_quu_mtq = 2 * legacy_settings.mtq_control_weight
    legacy_quu_rw = 2 * legacy_settings.rw_control_weight
    legacy_cond = max(legacy_quu_mtq, legacy_quu_rw) / min(legacy_quu_mtq, legacy_quu_rw)
    print(f"  Quu diag (MTQ): {legacy_quu_mtq:.0e}")
    print(f"  Quu diag (RW): {legacy_quu_rw:.0e}")
    print(f"  Estimated Quu condition: {legacy_cond:.0f}")
    
    # Normalized settings
    print("\n[Normalized Settings - Auto-Scaled]")
    norm_settings = create_normalized_settings(real_sat, verbose=True)
    
    # === Run trajectory tests ===
    print("\n" + "=" * 70)
    print("TRAJECTORY TESTS")
    print("=" * 70)
    
    results = []
    
    # Test 1: Legacy settings
    print("\n[1] Testing legacy (manual) settings...")
    result = run_trajectory_test(legacy_settings, real_sat, orb, x0, q_goal, tf, "Legacy")
    if result:
        results.append(result)
        print(f"  Traj gen time: {result['traj_time']:.2f}s")
        print(f"  Final error: {result['final_error']:.2f} deg")
        print(f"  RW sign change rate: {result['sign_change_rate']:.1f}%")
    
    # Test 2: Normalized settings
    print("\n[2] Testing normalized (auto-scaled) settings...")
    result = run_trajectory_test(norm_settings, real_sat, orb, x0, q_goal, tf, "Normalized")
    if result:
        results.append(result)
        print(f"  Traj gen time: {result['traj_time']:.2f}s")
        print(f"  Final error: {result['final_error']:.2f} deg")
        print(f"  RW sign change rate: {result['sign_change_rate']:.1f}%")
    
    # Test 3: MTQ+RW preset
    print("\n[3] Testing preset 'mtq_plus_rw'...")
    preset_settings = create_preset_settings(real_sat, 'mtq_plus_rw')
    result = run_trajectory_test(preset_settings, real_sat, orb, x0, q_goal, tf, "Preset")
    if result:
        results.append(result)
        print(f"  Traj gen time: {result['traj_time']:.2f}s")
        print(f"  Final error: {result['final_error']:.2f} deg")
        print(f"  RW sign change rate: {result['sign_change_rate']:.1f}%")
    
    # === Plot comparison ===
    if results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Normalized vs Legacy Settings Comparison', fontsize=14)
        
        colors = ['blue', 'green', 'orange', 'red']
        
        # Attitude error
        ax = axes[0, 0]
        for i, r in enumerate(results):
            ax.semilogy(r['time_hist'], r['att_error_hist'], color=colors[i], label=r['label'])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Attitude Error (deg)')
        ax.set_title('Attitude Error Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RW commands
        ax = axes[0, 1]
        for i, r in enumerate(results):
            ax.plot(r['controls'][3, :50] * 1000, color=colors[i], label=r['label'], alpha=0.7)
        ax.set_xlabel('Time step')
        ax.set_ylabel('RW Torque (mNm)')
        ax.set_title('RW Commands (first 50 steps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MTQ commands
        ax = axes[1, 0]
        for i, r in enumerate(results):
            ax.plot(r['controls'][0, :50], color=colors[i], label=f'{r["label"]} MTQ0', alpha=0.7)
        ax.set_xlabel('Time step')
        ax.set_ylabel('MTQ Dipole (Am²)')
        ax.set_title('MTQ0 Commands (first 50 steps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary bar chart
        ax = axes[1, 1]
        labels = [r['label'] for r in results]
        x = np.arange(len(labels))
        width = 0.25
        
        traj_times = [r['traj_time'] for r in results]
        final_errors = [r['final_error'] for r in results]
        sign_rates = [r['sign_change_rate'] for r in results]
        
        ax.bar(x - width, traj_times, width, label='Traj Time (s)', color='steelblue')
        ax.bar(x, final_errors, width, label='Final Error (deg)', color='coral')
        ax.bar(x + width, [s/10 for s in sign_rates], width, label='Sign Change Rate (%/10)', color='forestgreen')
        
        ax.set_ylabel('Value')
        ax.set_title('Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('examples/normalized_settings_comparison.png', dpi=150)
        print(f"\nSaved: examples/normalized_settings_comparison.png")
        plt.show()
    
    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<15} {'Traj Time':>12} {'Final Err':>12} {'Sign Change':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r['label']:<15} {r['traj_time']:>10.2f}s {r['final_error']:>10.2f}° {r['sign_change_rate']:>10.1f}%")


if __name__ == "__main__":
    main()
