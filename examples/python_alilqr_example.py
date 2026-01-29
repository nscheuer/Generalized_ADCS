"""
Example: Using PythonALILQR for step-by-step trajectory optimization analysis.

This script demonstrates how to use the Python-driven ALILQR optimizer
which wraps C++ subroutines but runs the outer loops in Python. This enables:

1. Real-time visualization of trajectory convergence
2. Analysis of constraint handling per iteration
3. Debugging augmented Lagrangian penalty evolution
4. Custom stopping criteria and analysis hooks

Run from the Generalized_ADCS directory:
    python examples/python_alilqr_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# Conditionally import visualization
try:
    from matplotlib.animation import FuncAnimation
    HAS_ANIMATION = True
except ImportError:
    HAS_ANIMATION = False


def setup_simple_example():
    """
    Set up a simple slew maneuver example.
    
    Returns the planner, settings, and initial conditions needed for optimization.
    """
    # Import after ensuring path is correct
    import sys
    sys.path.insert(0, '/home/pmckeen/Generalized_ADCS')
    
    from ADCS.controller.helpers.python_alilqr import PythonALILQR, IterationData, OptimizationResult
    from ADCS.controller.helpers import PlannerSettings
    from ADCS.controller.helpers.build_csat import build_cpp_satellite
    from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
    from ADCS.satellite_hardware.actuators import MTQ, RW
    from ADCS.satellite_hardware.errors.bias import Bias
    from ADCS.orbits.orbital_state import Orbital_State
    from ADCS.orbits.orbit import Orbit
    from ADCS.CONOPS.goallist import GoalList
    from ADCS.CONOPS.goals.inertial_pointing import InertialPointing
    from ADCS.orbits.universal_constants import TimeConstants
    
    import trajectory_planner.build.tplaunch as tplaunch
    
    # Create a simple satellite with MTQs and RWs
    J = np.diag([0.1, 0.1, 0.1])  # 10 kg, 0.1m cube inertia
    
    # 3 magnetorquers along body axes
    mtq_x = MTQ(axis=np.array([1, 0, 0]), max_torque=0.5, bias=Bias())  # 0.5 A*m^2 max dipole
    mtq_y = MTQ(axis=np.array([0, 1, 0]), max_torque=0.5, bias=Bias())
    mtq_z = MTQ(axis=np.array([0, 0, 1]), max_torque=0.5, bias=Bias())
    
    # 3 reaction wheels along body axes
    rw_x = RW(
        axis=np.array([1, 0, 0]), 
        max_torque=0.001,  # 1 mN*m max torque
        J=1e-5,  # wheel inertia
        h=np.array([0.0]),  # initial momentum
        h_max=np.array([0.01]),  # max momentum
        bias=Bias()
    )
    rw_y = RW(
        axis=np.array([0, 1, 0]), 
        max_torque=0.001,
        J=1e-5,
        h=np.array([0.0]),
        h_max=np.array([0.01]),
        bias=Bias()
    )
    rw_z = RW(
        axis=np.array([0, 0, 1]), 
        max_torque=0.001,
        J=1e-5,
        h=np.array([0.0]),
        h_max=np.array([0.01]),
        bias=Bias()
    )
    
    # Combine actuators (6 total: 3 MTQ + 3 RW)
    actuators = [mtq_x, mtq_y, mtq_z, rw_x, rw_y, rw_z]
    
    est_sat = EstimatedSatellite(
        J_0=J,
        actuators=actuators,
        sensors=[],
        boresight=np.array([0, 0, 1])
    )
    
    # Planner settings
    planner_settings = PlannerSettings(
        est_sat=est_sat,
        dt_control=1.0,
        dt_tvlqr=1.0,  # 1 second steps
        dt_tp=1.0,
        tvlqr_len=60,
        tvlqr_overlap=15,
        bdot_on=2,  # Use smart bdot for initial trajectory
    )
    
    # Build C++ satellite
    csat = build_cpp_satellite(est_sat=est_sat, planner_settings=planner_settings)
    
    # Create planner
    planner = tplaunch.Planner(
        csat,
        planner_settings.systemSettings(),
        planner_settings.mainAlilqrSettings(),
        planner_settings.secondAlilqrSettings(),
        planner_settings.initTrajSettings(),
        planner_settings.optMainCostSettings(),
        planner_settings.optSecondCostSettings(),
        planner_settings.optTVLQRCostSettings(tracking_LQR_formulation=0)
    )
    planner.setquaternionTo3VecMode(2)  # Full quaternion
    
    # Initial state: small angular velocity, identity quaternion
    # State for satellite with RWs includes wheel momenta
    omega_0 = np.array([0.05, -0.03, 0.02])  # rad/s
    q_0 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    h_rw_0 = np.array([0.0, 0.0, 0.0])  # Initial RW momenta
    x_0 = np.concatenate([omega_0, q_0, h_rw_0])
    
    # Create a simple orbit for environment vectors
    t_start = 0.01  # J2000 centuries
    duration = 30.0  # seconds
    
    # Initial orbital state (LEO)
    os_0 = Orbital_State.from_eci(
        r_eci=np.array([7000e3, 0, 0]),  # 7000 km
        v_eci=np.array([0, 7.5e3, 0]),   # ~7.5 km/s
        J2000=t_start
    )
    
    # Goal: point boresight at nadir (approximately -r direction)
    goal = InertialPointing(
        target_vector_eci=np.array([0, 0, 1]),  # Point +Z at some inertial direction
        target_w_eci=np.array([0, 0, 0])  # Zero angular velocity
    )
    goals = GoalList([goal])
    
    return (planner, planner_settings, x_0, os_0, t_start, duration, goals, est_sat)


def create_environment_vectors(planner_settings, est_sat, os_0, t_start, duration, goals):
    """Create environment vectors for the planner."""
    from ADCS.orbits.orbit import Orbit
    from ADCS.orbits.universal_constants import TimeConstants
    
    dt_seconds = planner_settings.dt_tvlqr
    N = int(np.ceil(duration / dt_seconds)) + 1
    t_end = t_start + duration * TimeConstants.sec2cent
    
    buffer_centuries = 10 * dt_seconds * TimeConstants.sec2cent
    t_end_buffered = t_end + buffer_centuries
    
    # Propagate orbit
    sim_orbit = Orbit(os0=os_0, end_time=t_end_buffered, dt=dt_seconds, use_J2=True, fast=True)
    sim_orbit.populate_environment(compute_B=True, compute_S=True)
    tp_orbit = sim_orbit.get_range(t_start, t_end, dt_seconds)
    
    orbit_data = tp_orbit.get_vecs()
    times = np.asarray(tp_orbit.times, dtype=np.float64)
    
    # Clip/pad times
    if len(times) > N:
        times = times[:N]
    elif len(times) < N:
        times = np.pad(times, (0, N - len(times)), mode="edge")
    
    # Get vectors
    R_raw, V_raw, B_raw, S_raw, Rho_raw = [np.asarray(d) for d in orbit_data]
    
    def to_3xN(x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] == 3:
            if x.shape[1] > N: return x[:, :N]
            if x.shape[1] < N: return np.pad(x, ((0,0),(0, N-x.shape[1])), mode="edge")
            return x
        if x.shape[1] == 3:
            x = x.T
            if x.shape[1] > N: return x[:, :N]
            if x.shape[1] < N: return np.pad(x, ((0,0),(0, N-x.shape[1])), mode="edge")
            return x
        return x
    
    R = np.asfortranarray(to_3xN(R_raw), dtype=np.float64)
    V = np.asfortranarray(to_3xN(V_raw), dtype=np.float64)
    B = np.asfortranarray(to_3xN(B_raw), dtype=np.float64)
    S = np.asfortranarray(to_3xN(S_raw), dtype=np.float64)
    rho = np.ascontiguousarray(Rho_raw.flatten()[:N], dtype=np.float64)
    if len(rho) < N:
        rho = np.pad(rho, (0, N - len(rho)), mode="edge")
    
    # Goal vectors (pointing at +Z inertial)
    E = np.zeros((3, N), dtype=np.float64, order='F')
    E[2, :] = 1.0  # Point at +Z
    
    # Satellite boresight
    A = np.zeros((3, N), dtype=np.float64, order='F')
    A[2, :] = 1.0  # Boresight is +Z body
    
    # Prop values
    p = np.zeros(N, dtype=np.float64)
    
    return (times, R, V, B, S, A, E, p, rho), N


def create_initial_trajectory(planner, x_0, N, dt):
    """Create an initial trajectory guess."""
    state_dim = len(x_0)
    ctrl_dim = 3  # magic actuators
    
    Xset = np.zeros((state_dim, N), dtype=np.float64)
    Uset = np.zeros((ctrl_dim, N), dtype=np.float64)
    times = np.arange(N, dtype=np.float64) * dt
    TQset = np.zeros((3, N), dtype=np.float64)
    
    # Initialize with constant state (will be refined by generateInitialTrajectory)
    for k in range(N):
        Xset[:, k] = x_0
    
    # Small random controls
    Uset = 0.0001 * (2 * np.random.random((ctrl_dim, N)) - 1)
    
    return (Xset, Uset, times, TQset)


def plot_iteration(iter_data, ax_traj=None, ax_cost=None, history=None):
    """Plot current iteration state."""
    if ax_traj is None or ax_cost is None:
        return
    
    # Clear axes
    ax_traj.clear()
    ax_cost.clear()
    
    # Plot trajectory (angular velocities)
    times = np.arange(iter_data.Xset.shape[1])
    ax_traj.plot(times, iter_data.Xset[0, :], 'r-', label='ωx')
    ax_traj.plot(times, iter_data.Xset[1, :], 'g-', label='ωy')
    ax_traj.plot(times, iter_data.Xset[2, :], 'b-', label='ωz')
    ax_traj.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_traj.set_xlabel('Time step')
    ax_traj.set_ylabel('Angular velocity (rad/s)')
    ax_traj.set_title(f'Outer: {iter_data.outer_iter}, Inner: {iter_data.inner_iter}')
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    
    # Plot cost history if available
    if history is not None and len(history) > 0:
        costs = [h.LA for h in history]
        cmaxes = [h.cmax for h in history]
        
        ax_cost.semilogy(costs, 'b-', label='Cost')
        ax_cost.semilogy(cmaxes, 'r-', label='Max Violation')
        ax_cost.set_xlabel('Iteration')
        ax_cost.set_ylabel('Value')
        ax_cost.set_title('Convergence')
        ax_cost.legend()
        ax_cost.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.pause(0.01)


def main():
    """Main example script."""
    print("="*70)
    print("Python ALILQR Example")
    print("="*70)
    
    # Setup
    print("\nSetting up problem...")
    planner, planner_settings, x_0, os_0, t_start, duration, goals, est_sat = setup_simple_example()
    
    # Create environment vectors
    print("Creating environment vectors...")
    vecs, N = create_environment_vectors(planner_settings, est_sat, os_0, t_start, duration, goals)
    
    # Create initial trajectory
    print("Creating initial trajectory...")
    dt = planner_settings.dt_tvlqr
    initial_traj = create_initial_trajectory(planner, x_0, N, dt)
    
    # Get settings
    cost_settings = planner_settings.optMainCostSettings()
    alilqr_settings = planner_settings.mainAlilqrSettings()
    
    # Import our Python ALILQR
    from ADCS.controller.helpers.python_alilqr import PythonALILQR, run_with_visualization
    
    # Create optimizer with verbose output
    py_alilqr = PythonALILQR(planner, verbose=True)
    
    # Option 1: Simple run with callback
    print("\n" + "="*70)
    print("Running optimization with step-by-step output...")
    print("="*70)
    
    history = []
    
    def my_callback(iter_data):
        history.append(iter_data)
        if len(history) % 5 == 0:
            print(f"  Progress: {len(history)} iterations, "
                  f"cost={iter_data.LA:.4e}, cmax={iter_data.cmax:.4e}")
    
    py_alilqr.debug_callback = my_callback
    
    result = py_alilqr.optimize(
        dt=dt,
        initial_traj=initial_traj,
        vecs=vecs,
        cost_settings=cost_settings,
        alilqr_settings=alilqr_settings,
        is_first_search=True
    )
    
    print(f"\nResult: success={result.success}")
    print(f"  Final cost: {result.final_cost:.6e}")
    print(f"  Final cmax: {result.final_cmax:.6e}")
    print(f"  Total iterations: {result.total_inner_iters}")
    print(f"  Outer iterations: {result.total_outer_iters}")
    
    # Plot final results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Angular velocities
    ax = axes[0, 0]
    times = np.arange(result.Xset.shape[1]) * dt
    ax.plot(times, result.Xset[0, :], 'r-', label='ωx')
    ax.plot(times, result.Xset[1, :], 'g-', label='ωy')
    ax.plot(times, result.Xset[2, :], 'b-', label='ωz')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular velocity (rad/s)')
    ax.set_title('Final Trajectory - Angular Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Quaternion
    ax = axes[0, 1]
    ax.plot(times, result.Xset[3, :], 'k-', label='q0')
    ax.plot(times, result.Xset[4, :], 'r-', label='q1')
    ax.plot(times, result.Xset[5, :], 'g-', label='q2')
    ax.plot(times, result.Xset[6, :], 'b-', label='q3')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Quaternion')
    ax.set_title('Final Trajectory - Attitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Controls
    ax = axes[1, 0]
    ax.plot(times[:-1], result.Uset[0, :-1], 'r-', label='u1')
    ax.plot(times[:-1], result.Uset[1, :-1], 'g-', label='u2')
    ax.plot(times[:-1], result.Uset[2, :-1], 'b-', label='u3')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control torque (Nm)')
    ax.set_title('Final Trajectory - Control')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convergence
    ax = axes[1, 1]
    costs = [h.LA for h in history]
    cmaxes = [h.cmax for h in history]
    iters = range(len(costs))
    ax.semilogy(iters, costs, 'b-', label='Cost (LA)')
    ax.semilogy(iters, cmaxes, 'r-', label='Max Violation')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Convergence History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/python_alilqr_result.png', dpi=150)
    print(f"\nPlot saved to /tmp/python_alilqr_result.png")
    
    # Option 2: Generator-based iteration for custom control
    print("\n" + "="*70)
    print("Demonstrating generator-based iteration (first 10 steps)...")
    print("="*70)
    
    initial_traj2 = create_initial_trajectory(planner, x_0, N, dt)
    py_alilqr2 = PythonALILQR(planner, verbose=False)
    
    for i, iter_data in enumerate(py_alilqr2.optimize_step_by_step(
        dt=dt,
        initial_traj=initial_traj2,
        vecs=vecs,
        cost_settings=cost_settings,
        alilqr_settings=alilqr_settings,
        is_first_search=True
    )):
        print(f"  Step {i}: outer={iter_data.outer_iter}, inner={iter_data.inner_iter}, "
              f"LA={iter_data.LA:.4e}, cmax={iter_data.cmax:.4e}")
        
        # Can do custom analysis here
        max_omega = np.max(np.abs(iter_data.Xset[:3, :]))
        max_u = np.max(np.abs(iter_data.Uset))
        print(f"    max(|ω|)={max_omega:.4f} rad/s, max(|u|)={max_u:.6f} Nm")
        
        if i >= 9:  # Stop after 10 iterations for demo
            print("  (stopping early for demo)")
            break
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    main()
