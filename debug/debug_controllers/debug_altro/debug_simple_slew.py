"""
Simple eigenaxis slew test - analytically known solution.

Problem: Rotate 10 degrees around z-axis in 10 seconds
- Start: q = [1,0,0,0], w = 0
- End: q = [cos(5deg), 0, 0, sin(5deg)], w = 0
- Single axis rotation -> torque should be smooth bang-bang around z

Expected optimal control (bang-bang):
- First ~5 steps: positive torque around z
- Last ~5 steps: negative torque around z (decelerate)
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track import PlannerSettings, DebugPlanner
from ADCS.controller.plan_and_track.build_csat import build_cpp_satellite
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize


def quat_from_axis_angle(axis, angle_rad):
    """Create quaternion from axis-angle."""
    axis = np.array(axis) / np.linalg.norm(axis)
    return np.array([np.cos(angle_rad/2),
                     axis[0]*np.sin(angle_rad/2),
                     axis[1]*np.sin(angle_rad/2),
                     axis[2]*np.sin(angle_rad/2)])


def quat_mult(q1, q2):
    """Quaternion multiplication q1 ⊗ q2. Format: [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def main():
    print("=" * 70)
    print("SIMPLE EIGENAXIS SLEW TEST")
    print("=" * 70)
    print("\nProblem: 10-degree rotation around Z-axis in 10 seconds")
    print("Expected: Bang-bang control - positive then negative torque around Z")

    np.random.seed(42)

    # Short horizon
    tf, dt = 10, 1.0
    N = int(tf / dt) + 1

    # Simple isotropic inertia for cleaner dynamics
    J = np.diagflat([0.1, 0.1, 0.1])  # Isotropic

    rw_max_torque = 0.01  # Larger torque for faster slew
    rws = [RW(axis=j, max_torque=rw_max_torque, J=0.001, h=0.0, h_max=0.1)
           for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    real_sat = Satellite(
        mass=1.0, J_0=J,
        actuators=rws, sensors=mtms, boresight=np.array([1, 0, 0])  # X-axis boresight
    )

    # Initial state: at rest, identity quaternion
    x0 = np.concatenate([np.zeros(3), normalize(np.array([1,0,0,0])), np.zeros(3)])

    # Target: rotate boresight [1,0,0] by 10 degrees around Z
    # After rotation, boresight points at [cos(10deg), sin(10deg), 0]
    angle_deg = 10
    angle_rad = angle_deg * np.pi / 180
    target_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])

    print(f"\nInitial boresight: [1, 0, 0]")
    print(f"Target direction:  [{target_dir[0]:.4f}, {target_dir[1]:.4f}, {target_dir[2]:.4f}]")
    print(f"Rotation angle: {angle_deg} degrees around Z-axis")

    # Fake orbital state (not used for this test)
    ephem = Ephemeris()
    os0 = Orbital_State(
        ephem=ephem, J2000=0.0,
        R=7000*np.array([1, 0, 0]), V=np.array([0, 7.5, 0]),
        B=np.array([0, 0, 0.00003]),  # Small B field
        S=np.array([1, 0, 0]), rho=0
    )

    # Planner settings - keep it simple
    planner_settings = PlannerSettings(est_sat=real_sat, bdot_on=0, dt_tp=dt)
    planner_settings.verbosity = False
    planner_settings.rw_stic_weight = 0  # Disable non-convex stiction
    planner_settings.rw_AM_weight = 0    # Disable AM cost
    planner_settings.rw_control_weight = 1e0  # Very low control cost -> should favor bang-bang
    planner_settings.cost_main.ang_vel = 1e0  # Low angular velocity cost
    planner_settings.cost_main.angle = 1e6    # Very high angle cost -> emphasize reaching target
    planner_settings.pass1.aug_lag.penalty_init = 1e2

    csat = build_cpp_satellite(est_sat=real_sat, planner_settings=planner_settings)

    planner = DebugPlanner(
        csat,
        planner_settings.systemSettings(),
        planner_settings.mainAlilqrSettings(),
        planner_settings.secondAlilqrSettings(),
        planner_settings.initTrajSettings(),
        planner_settings.optMainCostSettings(),
        planner_settings.optSecondCostSettings(),
        planner_settings.optTVLQRCostSettings(0),
        debug_level=0
    )
    planner.setquaternionTo3VecMode(2)

    # Environment vectors - constant for this simple test
    t_start = 0.0
    t_end = tf * TimeConstants.sec2cent

    times = np.linspace(t_start, t_end, N)
    # Use Fortran order (column-major) - matches Armadillo's storage
    R = np.tile(os0.R.reshape(3,1), (1, N)).astype(np.float64, order='F')
    V = np.tile(os0.V.reshape(3,1), (1, N)).astype(np.float64, order='F')
    B = np.tile(os0.B.reshape(3,1), (1, N)).astype(np.float64, order='F')
    S = np.tile(os0.S.reshape(3,1), (1, N)).astype(np.float64, order='F')
    rho = np.zeros(N, dtype=np.float64)

    # Goal: point boresight at target_dir (constant)
    A = np.tile(real_sat.get_boresight().reshape(3, 1), (1, N)).astype(np.float64, order='F')
    E = np.tile(target_dir.reshape(3,1), (1, N)).astype(np.float64, order='F')
    p = np.zeros(N, dtype=np.float64)

    vecsPy = (np.ascontiguousarray(times),
              np.asfortranarray(R), np.asfortranarray(V),
              np.asfortranarray(B), np.asfortranarray(S),
              np.asfortranarray(A), np.asfortranarray(E),
              np.ascontiguousarray(p), np.ascontiguousarray(rho))

    x0_clean = np.copy(x0.astype(np.float64).flatten(), order='C')
    u_limit = 0.75 * rw_max_torque

    print(f"\nControl limit: {u_limit} Nm")
    print(f"Timesteps: {N}")

    # Prepare and run
    (traj, vecs_dt, costSettings) = planner.prepareForAlilqr(
        vecsPy, dt, t_start, t_end, x0_clean, 0
    )

    auglagSettings = planner_settings.pass1.aug_lag.to_tuple()
    regSettings = planner_settings.pass1.regularization.to_tuple()
    lineSearchSettings = planner_settings.pass1.line_search.to_tuple()

    (Xset_orig, Uset_orig, Tk_orig, Tq_orig) = traj
    num_timesteps = Uset_orig.shape[1]

    # =========================================================================
    # INJECT KNOWN BANG-BANG TRAJECTORY
    # =========================================================================
    # For 10° rotation around Z in 10s with J=0.1 kg·m²:
    # Bang-bang torque: u = J * θ_f / t_switch² = 0.1 * 0.1745 / 25 = 0.000698 Nm
    # Switch at t=5s (k=5)

    u_opt = 0.000698  # Optimal bang-bang torque magnitude
    print(f"\n{'='*70}")
    print("INJECTING KNOWN BANG-BANG TRAJECTORY")
    print(f"{'='*70}")
    print(f"Optimal bang-bang torque: u = {u_opt:.6f} Nm")

    print(f"\nOriginal trajectory shapes:")
    print(f"  Xset: {Xset_orig.shape}, order={'F' if Xset_orig.flags['F_CONTIGUOUS'] else 'C'}")
    print(f"  Uset: {Uset_orig.shape}, order={'F' if Uset_orig.flags['F_CONTIGUOUS'] else 'C'}")
    print(f"  Tk: {Tk_orig.shape if hasattr(Tk_orig, 'shape') else type(Tk_orig)}")
    print(f"  x0_clean: {x0_clean}")

    # Build optimal control sequence - ALWAYS use Fortran order (C++ expects column-major)
    # Note: prepareForAlilqr returns C order but C++ functions expect Fortran order!
    Uset_bb = np.zeros(Uset_orig.shape, dtype=np.float64, order='F')
    for k in range(num_timesteps):
        if k < 5:  # First half: positive torque
            Uset_bb[2, k] = u_opt
        else:      # Second half: negative torque
            Uset_bb[2, k] = -u_opt

    # Build optimal state sequence by forward simulation
    # State: [w_x, w_y, w_z, q0, q1, q2, q3, h_x, h_y, h_z] (10 states for 3 RWs)
    Xset_bb = np.zeros(Xset_orig.shape, dtype=np.float64, order='F')
    Xset_bb[:, 0] = x0_clean  # Initial state

    for k in range(num_timesteps - 1):
        w = Xset_bb[0:3, k]
        q = Xset_bb[3:7, k]
        h = Xset_bb[7:10, k] if Xset_bb.shape[0] > 7 else np.zeros(3)

        # Control torque (only z-component)
        u_k = Uset_bb[:, k]

        # Simple Euler integration for isotropic inertia
        # w_dot = J^{-1} * (u - w x (J*w + h)) ≈ u/J for small w
        # For isotropic J=0.1*I: w_dot = 10*u
        w_new = w + dt * (10.0 * u_k[:3])

        # Quaternion update: q_dot = 0.5 * q ⊗ [0, w]
        w_quat = np.array([0, w[0], w[1], w[2]])
        q_dot = 0.5 * quat_mult(q, w_quat)
        q_new = q + dt * q_dot
        q_new = q_new / np.linalg.norm(q_new)  # Normalize

        # RW momentum: h_dot = -u (reaction)
        h_new = h - dt * u_k[:3]

        Xset_bb[0:3, k+1] = w_new
        Xset_bb[3:7, k+1] = q_new
        if Xset_bb.shape[0] > 7:
            Xset_bb[7:10, k+1] = h_new

    # Replace trajectory with bang-bang (ensure Fortran order - C++ expects column-major)
    Xset_bb = np.asfortranarray(Xset_bb)
    Uset_bb = np.asfortranarray(Uset_bb)
    traj = (Xset_bb, Uset_bb, Tk_orig, Tq_orig)

    print("\nInjected bang-bang trajectory:")
    print(f"{'k':>3} | {'u_z':>10} | {'w_z':>10} | {'q0':>10} | {'q3':>10} | {'h_z':>10}")
    print("-" * 70)
    for k in range(min(11, num_timesteps)):
        u_z = Uset_bb[2, k]
        w_z = Xset_bb[2, k]
        q0 = Xset_bb[3, k]
        q3 = Xset_bb[6, k]
        h_z = Xset_bb[9, k] if Xset_bb.shape[0] > 9 else 0
        print(f"{k:3d} | {u_z:10.6f} | {w_z:10.6f} | {q0:10.6f} | {q3:10.6f} | {h_z:10.6f}")

    test_auglag = (np.zeros((20, num_timesteps), order='F'), 1.0,
                   np.ones((20, num_timesteps), order='F'))
    (clist_test, cmax_test) = planner.maxViol(traj, vecs_dt, test_auglag)
    num_c = clist_test.shape[0]

    print(f"\nConstraint info:")
    print(f"  Number of constraints: {num_c}")
    print(f"  Max constraint violation: {cmax_test:.6e}")

    lam_init, lam_max, mu_init, mu_max, mu_scale = auglagSettings
    lambdas = np.zeros((num_c, num_timesteps), dtype=np.float64, order='F')
    mu = mu_init
    muk = mu * np.ones((num_c, num_timesteps), dtype=np.float64, order='F')
    auglag_vals = (lambdas, mu, muk)
    regs = (regSettings[0], regSettings[0])

    # Try computing the cost first
    cost_val = planner.cost2Func(traj, vecs_dt, auglag_vals, costSettings)
    print(f"  Cost of bang-bang trajectory: {cost_val:.6e}")

    # =========================================================================
    # FIRST: TEST WITH ORIGINAL TRAJECTORY (sanity check)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST: BACKWARD PASS ON ORIGINAL (random) TRAJECTORY")
    print("=" * 70)

    orig_traj = (np.asfortranarray(Xset_orig), np.asfortranarray(Uset_orig), Tk_orig, Tq_orig)
    try:
        (bp_results_orig, _) = planner.backwardPass(
            dt, orig_traj, vecs_dt, auglag_vals, regs, costSettings, regSettings, False
        )
        (Kset_orig, dset_orig, Sset_orig) = bp_results_orig
        print("Original trajectory backward pass: SUCCESS")
        print(f"  d norm: {np.linalg.norm(dset_orig):.6e}")
    except Exception as e:
        print(f"Original trajectory backward pass: FAILED - {e}")

    # =========================================================================
    # ANALYZE BACKWARD PASS ON KNOWN TRAJECTORY
    # =========================================================================
    print("\n" + "=" * 70)
    print("BACKWARD PASS ANALYSIS ON KNOWN BANG-BANG")
    print("=" * 70)

    (bp_results, new_regs) = planner.backwardPass(
        dt, traj, vecs_dt, auglag_vals, regs, costSettings, regSettings, False
    )

    (Kset, dset, Sset) = bp_results

    print(f"\nFeedforward term d (should be ~0 if trajectory is optimal):")
    print(f"{'k':>3} | {'d_x':>12} | {'d_y':>12} | {'d_z':>12}")
    print("-" * 55)
    for k in range(min(11, dset.shape[1])):
        print(f"{k:3d} | {dset[0,k]:12.6e} | {dset[1,k]:12.6e} | {dset[2,k]:12.6e}")

    print(f"\nIf d ≈ 0, the trajectory is locally optimal.")
    print(f"If d ≠ 0, it shows the direction the optimizer wants to move.")

    d_norm = np.linalg.norm(dset)
    print(f"\n||d|| = {d_norm:.6e}")

    # Run iterations
    print("\n" + "-" * 70)
    print("OPTIMIZATION (from bang-bang initial)")
    print("-" * 70)

    for it in range(20):
        (bp_results, regs) = planner.backwardPass(
            dt, traj, vecs_dt, auglag_vals, regs, costSettings, regSettings, False
        )
        (traj, cost, regs) = planner.forwardPass(
            dt, traj, vecs_dt, auglag_vals, bp_results, regs,
            costSettings, regSettings, lineSearchSettings, False
        )

    (Xset, Uset, _, _) = traj

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nControl trajectory (expect: u_z positive then negative):")
    print(f"{'k':>3} | {'u_x':>10} | {'u_y':>10} | {'u_z':>10} | {'w_z':>10}")
    print("-" * 55)
    for k in range(min(N, Uset.shape[1])):
        u_x, u_y, u_z = Uset[0,k], Uset[1,k], Uset[2,k]
        w_z = Xset[2,k] if k < Xset.shape[1] else 0
        print(f"{k:3d} | {u_x:10.6f} | {u_y:10.6f} | {u_z:10.6f} | {w_z:10.6f}")

    # Sign change analysis
    print(f"\nSign changes:")
    for ch, name in enumerate(['u_x', 'u_y', 'u_z']):
        u = Uset[ch, :]
        sc = np.sum(np.diff(np.sign(u)) != 0)
        print(f"  {name}: {sc} sign changes")

    # For eigenaxis slew around Z, we expect:
    # - u_x, u_y should be near zero
    # - u_z should have 1 sign change (bang-bang)
    print("\n" + "=" * 70)
    print("EXPECTED vs ACTUAL")
    print("=" * 70)
    print("For 10-deg Z-axis rotation:")
    print("  Expected u_x: ~0 (no sign changes)")
    print("  Expected u_y: ~0 (no sign changes)")
    print("  Expected u_z: bang-bang (1 sign change)")

    u_z = Uset[2, :]
    u_z_sc = np.sum(np.diff(np.sign(u_z)) != 0)
    if u_z_sc == 1:
        print("\n  ✓ Z-axis control is bang-bang!")
    else:
        print(f"\n  ✗ Z-axis has {u_z_sc} sign changes (expected 1)")


if __name__ == "__main__":
    main()
