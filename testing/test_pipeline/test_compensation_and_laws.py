"""
Unit tests for Phase 3: Compensation Full Suite + Sliding Mode Law.

Tests:
1. Frame rotation feedforward
2. Damping injection
3. Disturbance feedforward (gravity gradient)
4. Sliding mode law
5. Full compensator with all terms
6. Sliding mode vs existing Wisniewski controller
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ADCS.helpers.math_helpers import (
    normalize, rot_mat, quat_mult, quat_inv, norm, skewsym, Wmat,
)
from ADCS.pipeline.compensation.gyroscopic import compute_gyroscopic_torque
from ADCS.pipeline.compensation.frame_rotation import compute_frame_rotation_torque
from ADCS.pipeline.compensation.damping_injection import compute_damping_injection
from ADCS.pipeline.compensation.disturbance_ff import compute_disturbance_feedforward
from ADCS.pipeline.compensation.compensator import compensation_step
from ADCS.pipeline.control_law.sliding_mode_law import SlidingMode_Law
from ADCS.pipeline.control_law.pd_law import PD_Law
from ADCS.pipeline.data import (
    CompensationConfig, CompensationInputs, LawInterface,
)

PASS_COUNT = 0
FAIL_COUNT = 0


def check(condition, label):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {label}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {label}")


# ==================================================================
# Test 1: Gyroscopic torque
# ==================================================================
def test_gyroscopic():
    print("\n--- Gyroscopic compensation ---")

    J = np.diag([0.04, 0.06, 0.08])
    omega = np.array([0.01, -0.02, 0.03])
    h_rw = np.zeros(3)

    tau = compute_gyroscopic_torque(omega, J, h_rw)
    expected = np.cross(omega, J @ omega)
    check(norm(tau - expected) < 1e-14, "gyro: cross(w, Jw) with no RW")

    # With RW momentum
    h_rw = np.array([0.001, 0.0, -0.002])
    tau2 = compute_gyroscopic_torque(omega, J, h_rw)
    expected2 = np.cross(omega, J @ omega + h_rw)
    check(norm(tau2 - expected2) < 1e-14, "gyro: cross(w, Jw + h_rw)")

    # Zero omega => zero torque
    tau_zero = compute_gyroscopic_torque(np.zeros(3), J, h_rw)
    check(norm(tau_zero) < 1e-14, "gyro: zero omega => zero torque")


# ==================================================================
# Test 2: Frame rotation feedforward
# ==================================================================
def test_frame_rotation():
    print("\n--- Frame rotation feedforward ---")

    J = np.diag([0.04, 0.06, 0.08])
    dt = 1.0

    # Constant omega_ref => omega_ref_dot = 0
    w_ref = np.array([0.001, 0.0, 0.0])
    tau = compute_frame_rotation_torque(w_ref, w_ref, J, dt)
    expected = -np.cross(w_ref, J @ w_ref)  # J*0 - cross(w_ref, J*w_ref)
    check(norm(tau - expected) < 1e-14, "frame_rot: constant ref => only cross term")

    # Changing omega_ref
    w_ref_prev = np.array([0.001, 0.0, 0.0])
    w_ref_curr = np.array([0.002, 0.001, 0.0])
    tau2 = compute_frame_rotation_torque(w_ref_curr, w_ref_prev, J, dt)
    w_ref_dot = (w_ref_curr - w_ref_prev) / dt
    expected2 = J @ w_ref_dot - np.cross(w_ref_curr, J @ w_ref_curr)
    check(norm(tau2 - expected2) < 1e-14, "frame_rot: changing ref matches formula")

    # Zero reference => zero torque
    tau_zero = compute_frame_rotation_torque(np.zeros(3), np.zeros(3), J, dt)
    check(norm(tau_zero) < 1e-14, "frame_rot: zero ref => zero torque")


# ==================================================================
# Test 3: Damping injection
# ==================================================================
def test_damping_injection():
    print("\n--- Damping injection ---")

    omega = np.array([0.01, -0.02, 0.03])
    omega_ref = np.array([0.001, 0.0, 0.0])
    k_d = 0.5

    # Full attitude: P = I
    P_full = np.eye(3)
    tau = compute_damping_injection(omega, omega_ref, P_full, k_d)
    expected = -k_d * (omega - omega_ref)
    check(norm(tau - expected) < 1e-14, "damp: full attitude P=I")

    # Reduced attitude: P = I - b*b^T
    b = np.array([0.0, 0.0, 1.0])
    P_red = np.eye(3) - np.outer(b, b)
    tau_red = compute_damping_injection(omega, omega_ref, P_red, k_d)
    omega_err = omega - omega_ref
    expected_red = -k_d * (P_red @ omega_err)
    check(norm(tau_red - expected_red) < 1e-14, "damp: reduced P projects out b")
    check(abs(tau_red[2]) < 1e-14, "damp: z-axis damping is zero (boresight free)")

    # No goal: P = 0
    P_none = np.zeros((3, 3))
    tau_none = compute_damping_injection(omega, omega_ref, P_none, k_d)
    check(norm(tau_none) < 1e-14, "damp: no goal P=0 => zero torque")

    # Zero omega error => zero damping
    tau_eq = compute_damping_injection(omega_ref, omega_ref, P_full, k_d)
    check(norm(tau_eq) < 1e-14, "damp: at reference => zero")


# ==================================================================
# Test 4: Disturbance feedforward (gravity gradient)
# ==================================================================
def test_disturbance_ff():
    print("\n--- Disturbance feedforward ---")

    J = np.diag([0.04, 0.06, 0.08])
    q = np.array([1.0, 0.0, 0.0, 0.0])  # identity
    r_eci = 7000.0 * np.array([1.0, 0.0, 0.0])  # 7000 km along X

    tau = compute_disturbance_feedforward(q, r_eci, J)

    # At identity quaternion with r along body X:
    # r_hat_body = [1, 0, 0]
    # tau_gg = (3 mu / r^3) * cross([1,0,0], J @ [1,0,0])
    #        = (3 mu / r^3) * cross([1,0,0], [0.04, 0, 0])
    #        = 0
    # So feedforward should be ~0 for this case
    check(norm(tau) < 1e-14, "gg: aligned r and body X => zero gg torque")

    # Non-trivial case: 45 deg about Y (so r_body has X and Z components
    # => cross product with J @ r_body is nonzero for non-spherical J)
    q2 = normalize(np.array([np.cos(np.pi/8), 0, np.sin(np.pi/8), 0]))
    tau2 = compute_disturbance_feedforward(q2, r_eci, J)
    check(norm(tau2) > 0, "gg: rotated attitude => nonzero gg torque")

    # The result should be the NEGATIVE of the gravity gradient
    # (feedforward cancels the disturbance)
    R_b2i = rot_mat(q2)
    r_hat_body = R_b2i.T @ np.array([1.0, 0.0, 0.0])
    mu = 398600.4418
    r_mag = 7000.0
    tau_gg = (3 * mu / r_mag**3) * np.cross(r_hat_body, J @ r_hat_body)
    check(norm(tau2 + tau_gg) < 1e-14, "gg: feedforward = -tau_gg")

    # Disabled
    tau_disabled = compute_disturbance_feedforward(
        q2, r_eci, J, enable_gravity_gradient=False,
    )
    check(norm(tau_disabled) < 1e-14, "gg: disabled => zero")


# ==================================================================
# Test 5: Sliding mode law
# ==================================================================
def test_sliding_mode_law():
    print("\n--- Sliding mode law ---")

    J = np.diag([0.04, 0.06, 0.08])
    lq = 0.01 * np.eye(3)
    ls = 0.1 * np.eye(3)

    law = SlidingMode_Law(J, lq, ls)

    # Interface declarations
    check(law.interface.attitude_type == 'full', "SM: attitude_type = full")
    check(law.interface.includes_gyroscopic, "SM: includes gyroscopic")
    check(law.interface.includes_frame_rotation, "SM: includes frame rotation")

    # Zero error => zero torque
    q_err = np.zeros(3)
    w_err = np.zeros(3)
    tau_zero = law.compute(q_err, w_err, omega_raw=np.zeros(3), h_rw_body=np.zeros(3))
    check(norm(tau_zero) < 1e-14, "SM: zero error => zero torque")

    # Nonzero error
    q_err = np.array([0.01, -0.02, 0.005])
    w_err = np.array([0.002, -0.001, 0.003])
    omega_raw = np.array([0.003, -0.001, 0.003])
    h_rw = np.zeros(3)

    tau = law.compute(q_err, w_err, omega_raw=omega_raw, h_rw_body=h_rw)
    check(tau.shape == (3,), "SM: returns shape (3,)")
    check(norm(tau) > 0, "SM: nonzero error => nonzero torque")

    # Verify against hand-computed reference
    q0 = np.sqrt(max(0.0, 1.0 - np.dot(q_err, q_err)))
    q_err_full = np.array([q0, q_err[0], q_err[1], q_err[2]])
    q_err_dot = 0.5 * w_err @ Wmat(q_err_full).T

    s = J @ w_err + lq @ q_err
    tau_gyro = np.cross(omega_raw, J @ omega_raw + h_rw)
    tau_frame = J @ np.cross(omega_raw, w_err)
    tau_q_dot = lq @ q_err_dot[1:4]
    tau_slide = ls @ s
    expected = tau_gyro + tau_frame - tau_q_dot - tau_slide
    check(norm(tau - expected) < 1e-14, "SM: matches hand-computed formula")


# ==================================================================
# Test 6: Sliding mode vs existing Wisniewski
# ==================================================================
def test_sliding_mode_vs_wisniewski():
    print("\n--- Sliding mode vs Wisniewski formula ---")

    J = np.diag([0.04, 0.06, 0.08])
    lambda_q = 0.01 * np.eye(3)
    lambda_s = 0.1 * np.eye(3)

    sm_law = SlidingMode_Law(J, lambda_q, lambda_s)

    # Test state
    omega = np.array([0.01, -0.02, 0.03])
    q = normalize(np.array([0.8, 0.2, -0.3, 0.1]))
    q_goal = normalize(np.array([0.9, -0.1, 0.2, 0.3]))
    h_rw = np.array([0.001, 0.0, -0.002])

    # Compute error the same way as Wisniewski
    q_err = quat_mult(quat_inv(q_goal), q)
    if q_err[0] < 0:
        q_err = -q_err
    q_err_vec = q_err[1:4]

    omega_ref = np.zeros(3)  # ECI goal => zero ref
    R_b2i = rot_mat(q)
    omega_ref_body = R_b2i.T @ omega_ref
    w_err = omega - omega_ref_body

    # Pipeline sliding mode computation
    tau_pipeline = sm_law.compute(
        q_err_vec, w_err,
        omega_raw=omega, h_rw_body=h_rw,
    )

    # Wisniewski formula (manual computation)
    q0 = np.sqrt(max(0.0, 1.0 - np.dot(q_err_vec, q_err_vec)))
    q_err_full = np.array([q0, q_err_vec[0], q_err_vec[1], q_err_vec[2]])
    q_err_dot = 0.5 * w_err @ Wmat(q_err_full).T

    s = J @ w_err + lambda_q @ q_err_vec
    tau_gyro = np.cross(omega, J @ omega + h_rw)
    tau_frame = J @ np.cross(omega, w_err)
    tau_q_dot = lambda_q @ q_err_dot[1:4]
    tau_slide = lambda_s @ s
    tau_wisn = tau_gyro + tau_frame - tau_q_dot - tau_slide

    check(norm(tau_pipeline - tau_wisn) < 1e-14,
          "SM pipeline matches Wisniewski formula")

    # Verify auto-config disables gyro/frame compensation
    config = CompensationConfig.from_law_interface(sm_law.interface)
    check(not config.enable_gyroscopic,
          "SM auto-config: gyroscopic disabled (law handles it)")
    check(not config.enable_frame_rotation,
          "SM auto-config: frame_rotation disabled (law handles it)")


# ==================================================================
# Test 7: Full compensator integration
# ==================================================================
def test_full_compensator():
    print("\n--- Full compensator integration ---")

    J = np.diag([0.04, 0.06, 0.08])
    omega = np.array([0.01, -0.02, 0.03])
    h_rw = np.zeros(3)
    tau_law = np.array([-0.001, 0.002, -0.0005])

    # Gyroscopic only (Phase 1 behavior)
    config_gyro = CompensationConfig(
        enable_gyroscopic=True,
        enable_frame_rotation=False,
        enable_disturbance_ff=False,
        enable_damping_injection=False,
    )
    inputs = CompensationInputs(
        P=np.eye(3),
        omega_ref_body=np.zeros(3),
        goal_type='full',
    )
    tau1 = compensation_step(tau_law, omega, J, h_rw, config_gyro, inputs)
    expected1 = tau_law + np.cross(omega, J @ omega)
    check(norm(tau1 - expected1) < 1e-14, "comp: gyro only")

    # Add frame rotation
    w_ref_prev = np.array([0.001, 0.0, 0.0])
    w_ref_curr = np.array([0.002, 0.001, 0.0])
    inputs2 = CompensationInputs(
        P=np.eye(3),
        omega_ref_body=w_ref_curr,
        goal_type='full',
    )
    config_fr = CompensationConfig(
        enable_gyroscopic=True,
        enable_frame_rotation=True,
    )
    tau2 = compensation_step(
        tau_law, omega, J, h_rw, config_fr, inputs2,
        omega_ref_body_prev=w_ref_prev, dt=1.0,
    )
    tau_fr = compute_frame_rotation_torque(w_ref_curr, w_ref_prev, J, 1.0)
    expected2 = expected1 + tau_fr
    check(norm(tau2 - expected2) < 1e-14, "comp: gyro + frame_rotation")

    # Add damping injection
    config_damp = CompensationConfig(
        enable_gyroscopic=False,
        enable_damping_injection=True,
        damping_gain=0.5,
    )
    inputs_damp = CompensationInputs(
        P=np.eye(3),
        omega_ref_body=np.zeros(3),
        goal_type='full',
        inject_damping=True,
    )
    tau3 = compensation_step(tau_law, omega, J, h_rw, config_damp, inputs_damp)
    expected3 = tau_law + compute_damping_injection(omega, np.zeros(3), np.eye(3), 0.5)
    check(norm(tau3 - expected3) < 1e-14, "comp: damping injection only")

    # Auto-config from PD law
    pd_law = PD_Law(kp=1.0, kd=0.5, eps=0.1)
    config_auto = CompensationConfig.from_law_interface(pd_law.interface)
    check(config_auto.enable_gyroscopic, "auto-config PD: gyro enabled")
    check(not config_auto.enable_disturbance_ff, "auto-config PD: disturbance FF disabled")

    # Auto-config from Sliding mode law
    sm_law = SlidingMode_Law(J, 0.01*np.eye(3), 0.1*np.eye(3))
    config_sm = CompensationConfig.from_law_interface(sm_law.interface)
    check(not config_sm.enable_gyroscopic, "auto-config SM: gyro disabled (law includes)")
    check(not config_sm.enable_frame_rotation, "auto-config SM: frame_rot disabled (law includes)")


# ==================================================================
# Main
# ==================================================================
def main():
    print("=" * 60)
    print("Phase 3: Compensation & Control Law Tests")
    print("=" * 60)

    test_gyroscopic()
    test_frame_rotation()
    test_damping_injection()
    test_disturbance_ff()
    test_sliding_mode_law()
    test_sliding_mode_vs_wisniewski()
    test_full_compensator()

    print("\n" + "=" * 60)
    print(f"Results: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    if FAIL_COUNT == 0:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    return 0 if FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
