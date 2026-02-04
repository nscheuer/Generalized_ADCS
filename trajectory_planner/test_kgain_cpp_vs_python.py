#!/usr/bin/env python3
"""
Compare C++ and Python K-gain warm-start implementations.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy.linalg import norm

# Import Python implementation FIRST (which internally imports tplaunch)
from ADCS.controller.helpers.mtq_warm_start import kgain_warm_start_controls
from ADCS.helpers.math_helpers import quat_mult, quat_inv, normalize

# Access tplaunch and pysat through the already-imported module path
import trajectory_planner.build.tplaunch as tp
import trajectory_planner.build.pysat as pysat


def create_test_trajectory(n_state=8, n_ctrl=4, N=11, dt=10.0):
    """Create a simple test trajectory with known dynamics."""
    tf = (N - 1) * dt

    # Simple rotation from identity to 45° about z-axis
    Xset = np.zeros((n_state, N))
    for k in range(N):
        t = k * dt
        angle = (t / tf) * np.pi / 4  # 45 degrees over trajectory
        Xset[0:3, k] = [0, 0, 0.01]  # Small constant omega_z
        Xset[3, k] = np.cos(angle / 2)  # q0
        Xset[6, k] = np.sin(angle / 2)  # q3 (z rotation)
        if n_state > 7:
            Xset[7, k] = 0.0001  # Small RW momentum

    # Small MTQ controls
    Uset = np.zeros((n_ctrl, N))
    Uset[0:3, :] = 0.01 * np.random.randn(3, N)  # Small random MTQ
    if n_ctrl > 3:
        Uset[3, :] = 1e-6 * np.random.randn(N)  # Tiny RW torque

    return Xset, Uset, tf


def create_simple_kgains(n_ctrl, n_reduced, N):
    """Create simple diagonal-ish K-gains for testing."""
    # K-gains that provide mild feedback
    Kset = np.zeros((n_ctrl, n_reduced, N - 1))
    for k in range(N - 1):
        # Small diagonal gains
        for i in range(min(n_ctrl, n_reduced)):
            Kset[i, i, k] = 0.1 * (1 + 0.01 * k)
    return Kset


def flatten_kgains(Kset_3d):
    """Flatten K-gains to match C++ packageK format."""
    n_ctrl, n_reduced, N = Kset_3d.shape
    Kset_flat = np.zeros((n_ctrl * n_reduced, N))
    for k in range(N):
        for row in range(n_ctrl):
            for col in range(n_reduced):
                flat_idx = row * n_reduced + col
                Kset_flat[flat_idx, k] = Kset_3d[row, col, k]
    return Kset_flat


def run_python_kgain(Xset, Uset, Kset_flat, dt_coarse, dt_fine, tf):
    """Run Python K-gain warm-start."""
    # Simple dynamics for Python (just quaternion kinematics)
    def simple_dynamics(x, u, dt, k=0):
        w = x[0:3]
        q = x[3:7]

        # Quaternion kinematics
        w_quat = np.array([0, w[0], w[1], w[2]])
        q_dot = 0.5 * quat_mult(q, w_quat)

        x_next = x.copy()
        x_next[3:7] = normalize(q + q_dot * dt)
        return x_next

    # Run Python K-gain warm-start
    Xset_py, Uset_py = kgain_warm_start_controls(
        Xset_coarse=Xset,
        Uset_coarse=Uset,
        Kset_coarse=Kset_flat,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
        tf=tf,
        dynamics_func=simple_dynamics,
        quat_to_3vec_mode=2,  # Cayley
        verbose=False
    )

    return Xset_py, Uset_py


def run_cpp_kgain(Xset, Uset, Kset_flat, dt_coarse, dt_fine, tf, sat):
    """Run C++ K-gain warm-start via pybind11."""
    N_fine = int(tf / dt_fine) + 1

    # Create vecs_fine (VECTOR_INFO_FORM)
    t_fine = np.linspace(0, tf, N_fine)
    Rset = np.zeros((3, N_fine))
    Rset[0, :] = 7000e3  # 7000 km altitude
    Vset = np.zeros((3, N_fine))
    Vset[1, :] = 7.5e3  # Orbital velocity
    Bset = np.zeros((3, N_fine))
    Bset[2, :] = 30e-6  # 30 µT B-field
    Sset = np.zeros((3, N_fine))
    Sset[0, :] = 1.0  # Sun along x
    satvec = np.zeros((3, N_fine))
    satvec[2, :] = 1.0
    ECIvec = np.zeros((3, N_fine))
    ECIvec[2, :] = 1.0
    pset = np.zeros(N_fine)
    rhovec = np.zeros(N_fine)

    # Call C++ function
    Xset_cpp, Uset_cpp, t_out, TQset_cpp = tp.kgain_warm_start(
        Xset, Uset, Kset_flat,
        dt_coarse, dt_fine, tf,
        sat,
        t_fine, Rset, Vset, Bset, Sset, satvec, ECIvec, pset, rhovec,
        2  # quat_to_3vec_mode = Cayley
    )

    return Xset_cpp, Uset_cpp


def main():
    print("=" * 60)
    print("C++ vs Python K-gain Warm-Start Comparison")
    print("=" * 60)

    # Test parameters
    dt_coarse = 10.0
    dt_fine = 2.0
    N_coarse = 11
    tf = (N_coarse - 1) * dt_coarse
    N_fine = int(tf / dt_fine) + 1

    n_state = 7  # MTQ-only: omega(3) + quat(4)
    n_ctrl = 3   # 3 MTQ
    n_reduced = 6  # omega_err(3) + quat_err(3)
    n_rw = 0

    print(f"\nTest setup:")
    print(f"  dt_coarse={dt_coarse}s, dt_fine={dt_fine}s, tf={tf}s")
    print(f"  N_coarse={N_coarse}, N_fine={N_fine}")
    print(f"  n_state={n_state}, n_ctrl={n_ctrl}, n_reduced={n_reduced}")

    # Create test data
    np.random.seed(42)
    Xset, Uset, _ = create_test_trajectory(n_state, n_ctrl, N_coarse, dt_coarse)
    Kset_3d = create_simple_kgains(n_ctrl, n_reduced, N_coarse)
    Kset_flat = flatten_kgains(Kset_3d)

    print(f"\nInput shapes:")
    print(f"  Xset: {Xset.shape}")
    print(f"  Uset: {Uset.shape}")
    print(f"  Kset_flat: {Kset_flat.shape}")

    # Run Python version
    print("\nRunning Python K-gain warm-start...")
    try:
        Xset_py, Uset_py = run_python_kgain(Xset, Uset, Kset_flat, dt_coarse, dt_fine, tf)
        print(f"  Python output: X={Xset_py.shape}, U={Uset_py.shape}")
    except Exception as e:
        print(f"  Python error: {e}")
        Xset_py, Uset_py = None, None

    # Run C++ version
    print("\nRunning C++ K-gain warm-start...")
    try:
        # Create satellite with 3 MTQs
        sat = pysat.Satellite()
        sat.change_Jcom(np.diag([0.05, 0.05, 0.05]))
        # Add 3 MTQs along x, y, z axes (axis, max_moment, min_moment)
        m_max = 0.2  # A·m²
        sat.add_MTQ(np.array([1.0, 0.0, 0.0]), m_max, -m_max)
        sat.add_MTQ(np.array([0.0, 1.0, 0.0]), m_max, -m_max)
        sat.add_MTQ(np.array([0.0, 0.0, 1.0]), m_max, -m_max)

        Xset_cpp, Uset_cpp = run_cpp_kgain(Xset, Uset, Kset_flat, dt_coarse, dt_fine, tf, sat)
        print(f"  C++ output: X={Xset_cpp.shape}, U={Uset_cpp.shape}")
    except Exception as e:
        print(f"  C++ error: {e}")
        import traceback
        traceback.print_exc()
        Xset_cpp, Uset_cpp = None, None

    # Compare results
    if Xset_py is not None and Xset_cpp is not None:
        print("\n" + "=" * 60)
        print("Comparison Results")
        print("=" * 60)

        # Ensure same shape
        min_cols = min(Xset_py.shape[1], Xset_cpp.shape[1])

        # State comparison
        X_diff = Xset_py[:, :min_cols] - Xset_cpp[:, :min_cols]
        print(f"\nState difference (X):")
        print(f"  Max |diff|: {np.max(np.abs(X_diff)):.6e}")
        print(f"  Mean |diff|: {np.mean(np.abs(X_diff)):.6e}")
        print(f"  RMS diff: {np.sqrt(np.mean(X_diff**2)):.6e}")

        # Quaternion angle difference at end
        q_py_end = Xset_py[3:7, -1]
        q_cpp_end = Xset_cpp[3:7, -1]
        q_err = quat_mult(quat_inv(q_py_end), q_cpp_end)
        angle_diff = 2 * np.arccos(np.clip(np.abs(q_err[0]), 0, 1)) * 180 / np.pi
        print(f"\nFinal quaternion angle difference: {angle_diff:.4f}°")

        # Control comparison
        min_cols_u = min(Uset_py.shape[1], Uset_cpp.shape[1])
        U_diff = Uset_py[:, :min_cols_u] - Uset_cpp[:, :min_cols_u]
        print(f"\nControl difference (U):")
        print(f"  Max |diff|: {np.max(np.abs(U_diff)):.6e}")
        print(f"  Mean |diff|: {np.mean(np.abs(U_diff)):.6e}")

        # Verdict
        x_match = np.max(np.abs(X_diff)) < 1e-4
        u_match = np.max(np.abs(U_diff)) < 1e-4

        print("\n" + "=" * 60)
        if x_match and u_match:
            print("✅ C++ and Python implementations MATCH!")
        else:
            print("❌ Implementations DIFFER - review needed")
            if not x_match:
                print(f"   State mismatch: max diff = {np.max(np.abs(X_diff)):.6e}")
            if not u_match:
                print(f"   Control mismatch: max diff = {np.max(np.abs(U_diff)):.6e}")
        print("=" * 60)
    else:
        print("\n❌ Could not compare - one or both implementations failed")


if __name__ == "__main__":
    main()
