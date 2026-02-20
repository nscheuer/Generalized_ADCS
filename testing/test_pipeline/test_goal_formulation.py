"""
Unit tests for Phase 2 Goal Formulation components.

Tests quaternion set parameterization, attitude error conversions,
omega reference computation, convention conversion, and the full
goal_formulation_step with GoalSpec inputs.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ADCS.helpers.math_helpers import (
    normalize, rot_mat, quat_mult, quat_inv, quat_diff, norm,
)
from ADCS.pipeline.goal_formulation.quat_set import (
    compute_set_basis, select_nearest_quaternion, find_perpendicular,
)
from ADCS.pipeline.goal_formulation.conventions import (
    convert_quat_convention, quat_conjugate,
)
from ADCS.pipeline.goal_formulation.attitude_error import (
    attitude_full_to_full,
    attitude_reduced_to_full,
    attitude_reduced_to_reduced,
    attitude_none,
    convert_error_representation,
    zero_attitude,
    AlternatingState,
)
from ADCS.pipeline.goal_formulation.omega_ref import compute_omega_ref_world
from ADCS.pipeline.goal_formulation.goal_formulation import goal_formulation_step
from ADCS.pipeline.data import GoalSpec, WorldVectorSpec, LawInterface

# Reusable orbital state mock
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants


def _make_os():
    """Create a test orbital state."""
    ephem = Ephemeris()
    R0 = 7000 * np.array([0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2])
    V0 = np.array([8.0, 0.0, 0.0])
    return Orbital_State(ephem=ephem, J2000=0.22, R=R0, V=V0)


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
# Test 1: find_perpendicular
# ==================================================================
def test_find_perpendicular():
    print("\n--- find_perpendicular ---")
    for v in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
              normalize(np.array([1, 1, 1])), normalize(np.array([-0.3, 0.7, 0.2]))]:
        p = find_perpendicular(v)
        check(abs(np.dot(v, p)) < 1e-12, f"perp to {v}: dot={np.dot(v, p):.2e}")
        check(abs(norm(p) - 1.0) < 1e-12, f"  unit norm: {norm(p):.12f}")


# ==================================================================
# Test 2: Quaternion set basis
# ==================================================================
def test_quat_set_basis():
    print("\n--- compute_set_basis ---")

    # Case 1: 90-degree rotation (b_hat=[0,0,1], u_hat=[1,0,0])
    b = np.array([0.0, 0.0, 1.0])
    u = np.array([1.0, 0.0, 0.0])
    x_bar, y_bar = compute_set_basis(b, u)

    check(abs(norm(x_bar) - 1.0) < 1e-10, "x_bar is unit quaternion")
    check(abs(y_bar[0]) < 1e-10, "y_bar scalar is 0 (pure quaternion)")
    check(abs(norm(y_bar[1:]) - 1.0) < 1e-10, "y_bar vector part is unit")

    # Verify: applying x_bar rotates b to u (at beta=0)
    R_x = rot_mat(x_bar)
    b_rotated = R_x @ b
    check(norm(b_rotated - u) < 1e-10, "x_bar rotates b to u")

    # Case 2: Nearly anti-parallel (regularization test)
    b2 = np.array([0.0, 0.0, 1.0])
    u2 = np.array([0.0, 0.0, -1.0])  # anti-parallel
    x_bar2, y_bar2 = compute_set_basis(b2, u2, epsilon_reg=1e-6)
    check(not np.any(np.isnan(x_bar2)), "anti-parallel: no NaN in x_bar")
    check(not np.any(np.isnan(y_bar2)), "anti-parallel: no NaN in y_bar")

    # Any quaternion from the set should rotate b to u (180 deg)
    R2 = rot_mat(x_bar2)
    b2_rotated = R2 @ b2
    check(norm(b2_rotated - u2) < 1e-4, "anti-parallel: x_bar rotates b to -b")

    # Case 3: Already aligned (theta ~= 0)
    b3 = np.array([1.0, 0.0, 0.0])
    u3 = np.array([1.0, 0.0, 0.0])
    x_bar3, y_bar3 = compute_set_basis(b3, u3)
    check(abs(x_bar3[0] - 1.0) < 1e-10, "aligned: x_bar ≈ identity quaternion")


# ==================================================================
# Test 3: Nearest quaternion selection
# ==================================================================
def test_nearest_quaternion():
    print("\n--- select_nearest_quaternion ---")

    b = np.array([0.0, 0.0, 1.0])
    u = np.array([1.0, 0.0, 0.0])
    x_bar, y_bar = compute_set_basis(b, u)

    # Current attitude = identity
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_g = select_nearest_quaternion(x_bar, y_bar, q)

    # q_g should be a valid unit quaternion
    check(abs(norm(q_g) - 1.0) < 1e-10, "q_g is unit quaternion")

    # q_g should rotate b to u
    R_g = rot_mat(q_g)
    b_rotated = R_g @ b
    check(norm(b_rotated - u) < 1e-10, "q_g rotates b to u")

    # q_g should be close to q (nearest from the set)
    # Since q = identity, and the set contains the 90-deg rotation,
    # geodesic distance should be minimal among set members
    q_e = quat_diff(q, q_g)
    angle_err = 2.0 * np.arccos(np.clip(abs(q_e[0]), 0, 1))
    check(angle_err < np.pi, f"geodesic angle: {np.degrees(angle_err):.1f} deg")

    # Test with a different current attitude (rotated around z by 45 deg)
    q2 = normalize(np.array([np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)]))
    q_g2 = select_nearest_quaternion(x_bar, y_bar, q2)
    R_g2 = rot_mat(q_g2)
    b_rotated2 = R_g2 @ b
    check(norm(b_rotated2 - u) < 1e-10, "q_g2 rotates b to u (different q)")


# ==================================================================
# Test 4: Convention conversion
# ==================================================================
def test_conventions():
    print("\n--- Convention conversion ---")

    q_e = normalize(np.array([0.9, 0.1, -0.2, 0.3]))

    # Identity conversion (same convention in and out)
    law_id = LawInterface(error_convention='goal_times_current_inv')
    q_out = convert_quat_convention(q_e, 'hamilton_scalar_first', law_id)
    check(norm(q_out - q_e) < 1e-12, "identity conversion")

    # Scalar-last (same error convention, only storage order changes)
    law_sl = LawInterface(quat_convention='hamilton_scalar_last',
                          error_convention='goal_times_current_inv')
    q_sl = convert_quat_convention(q_e, 'hamilton_scalar_first', law_sl)
    expected_sl = np.array([q_e[1], q_e[2], q_e[3], q_e[0]])
    check(norm(q_sl - expected_sl) < 1e-12, "scalar-last reorder")

    # Error convention flip
    law_flip = LawInterface(error_convention='current_inv_times_goal')
    q_flip = convert_quat_convention(q_e, 'hamilton_scalar_first', law_flip)
    expected_flip = quat_conjugate(q_e)
    check(norm(q_flip - expected_flip) < 1e-12, "error convention flip = conjugate")

    # Round-trip: convert to scalar-last, then back
    # (would need inverse function, but verify structure)
    check(abs(q_sl[3] - q_e[0]) < 1e-12, "scalar moved to last position")


# ==================================================================
# Test 5: Attitude error — full to full
# ==================================================================
def test_attitude_full_to_full():
    print("\n--- attitude_full_to_full ---")

    q = normalize(np.array([0.8, 0.2, -0.3, 0.1]))  # current
    q_g = normalize(np.array([0.9, -0.1, 0.2, 0.3]))  # goal

    # Convention: q_e = q_g^{-1} * q, return vector part [1:4]
    law = LawInterface()
    q_e = attitude_full_to_full(q_g, q, law)

    # Should match existing Attitude_Goal convention: q_err = q_ref^{-1} * q
    q_e_existing = quat_mult(quat_inv(q_g), q)
    if q_e_existing[0] < 0:
        q_e_existing = -q_e_existing

    check(q_e.shape == (3,), "full_to_full returns 3-vector")
    check(
        norm(q_e - q_e_existing[1:4]) < 1e-10,
        "full_to_full matches existing Attitude_Goal convention"
    )

    # Zero error case
    q_e_zero = attitude_full_to_full(q, q, law)
    check(norm(q_e_zero) < 1e-10, "zero error: vector ≈ 0")


# ==================================================================
# Test 6: Attitude error — reduced to full
# ==================================================================
def test_attitude_reduced_to_full():
    print("\n--- attitude_reduced_to_full ---")

    b = np.array([0.0, 0.0, 1.0])  # body Z
    u = np.array([1.0, 0.0, 0.0])  # world X

    # Current attitude: identity (body Z points at ECI Z)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    law = LawInterface(error_convention='current_inv_times_goal')

    q_e = attitude_reduced_to_full(b, u, q, law)

    # q_e should be nonzero 3-vector (body Z != world X at identity)
    check(q_e.shape == (3,), "reduced_to_full returns 3-vector")
    check(norm(q_e) > 0.01, "nonzero error for misaligned vectors")

    # If we apply q_g = q_e (in goal_times_current_inv) to the body,
    # b should align with u
    # q_e here is in current_inv_times_goal convention, so q_g * q^{-1} was conjugated
    # Let's just verify the reduced constraint: R(q_g) @ b ≈ u
    # where q_g = select_nearest_quaternion(...)
    from ADCS.pipeline.goal_formulation.quat_set import compute_set_basis, select_nearest_quaternion
    x_bar, y_bar = compute_set_basis(b, u)
    q_g = select_nearest_quaternion(x_bar, y_bar, q)
    R_g = rot_mat(q_g)
    check(norm(R_g @ b - u) < 1e-10, "selected q_g aligns b with u")


# ==================================================================
# Test 7: Reduced to reduced
# ==================================================================
def test_attitude_reduced_to_reduced():
    print("\n--- attitude_reduced_to_reduced ---")

    b = np.array([0.0, 0.0, 1.0])
    u = normalize(np.array([1.0, 1.0, 0.0]))
    q = normalize(np.array([0.8, 0.2, -0.3, 0.1]))

    # Body frame output
    law_body = LawInterface(attitude_type='reduced', world_vector_frame='body')
    b_out, r_body = attitude_reduced_to_reduced(b, u, q, law_body)
    check(norm(b_out - b) < 1e-12, "b_hat passed through")
    R = rot_mat(q)
    check(norm(r_body - R.T @ u) < 1e-12, "u transformed to body frame")

    # World frame output
    law_world = LawInterface(attitude_type='reduced', world_vector_frame='world')
    b_out2, u_out = attitude_reduced_to_reduced(b, u, q, law_world)
    check(norm(u_out - u) < 1e-12, "u passed through in world frame")


# ==================================================================
# Test 8: Projection matrix P
# ==================================================================
def test_projection_matrix():
    print("\n--- Projection matrix P ---")

    b = normalize(np.array([1.0, 0.0, 0.0]))
    P = np.eye(3) - np.outer(b, b)

    # P should project out the b component
    v = np.array([3.0, 4.0, 5.0])
    Pv = P @ v
    check(abs(np.dot(Pv, b)) < 1e-12, "P projects out b component")
    check(abs(Pv[0]) < 1e-12, "b=[1,0,0]: x component zeroed")
    check(abs(Pv[1] - 4.0) < 1e-12, "y component preserved")
    check(abs(Pv[2] - 5.0) < 1e-12, "z component preserved")

    # P is idempotent: P @ P = P
    PP = P @ P
    check(norm(PP - P) < 1e-12, "P is idempotent (P² = P)")

    # P is symmetric
    check(norm(P - P.T) < 1e-12, "P is symmetric")


# ==================================================================
# Test 9: Analytical omega_ref for nadir
# ==================================================================
def test_omega_ref_analytical():
    print("\n--- omega_ref analytical ---")

    os = _make_os()
    r = np.asarray(os.R).flatten()
    v = np.asarray(os.V).flatten()

    # Expected: (r x v) / |r|^2
    expected = np.cross(r, v) / np.dot(r, r)

    goal_spec = GoalSpec(
        goal_type='reduced',
        b_hat=np.array([0, 0, 1]),
        u_spec=WorldVectorSpec(type='named', name='nadir'),
    )

    omega_ref = compute_omega_ref_world(goal_spec, None, 'reduced', np.array([1,0,0,0.]), os)
    check(norm(omega_ref - expected) < 1e-12, f"nadir omega_ref matches (r×v)/|r|²")

    # For 'none' goal type
    goal_none = GoalSpec(goal_type='none')
    omega_none = compute_omega_ref_world(goal_none, None, 'none', np.array([1,0,0,0.]), os)
    check(norm(omega_none) < 1e-12, "none goal: omega_ref = 0")


# ==================================================================
# Test 10: Full goal_formulation_step with GoalSpec
# ==================================================================
def test_full_goal_formulation_step():
    print("\n--- Full goal_formulation_step ---")

    os = _make_os()
    q = normalize(np.array([0.8, 0.2, -0.3, 0.1]))
    omega = np.array([0.01, -0.005, 0.008])

    # Test 10a: Full goal with full law
    q_goal = normalize(np.array([0.9, -0.1, 0.2, 0.3]))
    spec_full = GoalSpec(goal_type='full', q_goal=q_goal)
    law_full = LawInterface(attitude_type='full', omega_type='omega_error',
                            error_convention='current_inv_times_goal')

    gf = goal_formulation_step(spec_full, q, omega, os, law_full)
    check(gf.goal_type == 'full', "full goal: type='full'")
    check(norm(gf.P - np.eye(3)) < 1e-12, "full goal: P = I")
    check(gf.omega_output is not None, "full goal: omega_output not None")
    check(gf.attitude_output.shape == (3,), "full goal: attitude is 3-vec (error)")

    # Test 10b: Reduced goal with full law (quaternion set selection)
    spec_red = GoalSpec(
        goal_type='reduced',
        b_hat=np.array([0, 0, 1]),
        u_spec=WorldVectorSpec(type='named', name='nadir'),
    )
    gf_red = goal_formulation_step(spec_red, q, omega, os, law_full)
    check(gf_red.goal_type == 'reduced', "reduced goal: type='reduced'")
    P_red = gf_red.P
    b = np.array([0, 0, 1.0])
    check(norm(P_red @ b) < 1e-10, "reduced goal: P projects out b")
    check(gf_red.attitude_output.shape == (3,), "reduced->full: attitude is 3-vec")

    # Test 10c: None goal
    spec_none = GoalSpec(goal_type='none')
    gf_none = goal_formulation_step(spec_none, q, omega, os, law_full)
    check(gf_none.goal_type == 'none', "none goal: type='none'")
    check(norm(gf_none.P) < 1e-12, "none goal: P = 0")
    check(norm(gf_none.attitude_output) < 1e-10, "none goal: zero error")

    # Test 10d: Reduced goal with reduced law
    law_red = LawInterface(attitude_type='reduced', omega_type='omega_error',
                           world_vector_frame='body')
    gf_rr = goal_formulation_step(spec_red, q, omega, os, law_red)
    check(isinstance(gf_rr.attitude_output, tuple), "reduced->reduced: tuple output")
    b_out, r_target = gf_rr.attitude_output
    check(norm(b_out - np.array([0, 0, 1])) < 1e-12, "reduced->reduced: b passed through")


# ==================================================================
# Test 11: Attitude error representations
# ==================================================================
def test_attitude_representations():
    print("\n--- Attitude error representations ---")
    from ADCS.helpers.math_helpers import quat_to_mrp, quat_to_cayley, quat_to_euler

    q = normalize(np.array([0.8, 0.2, -0.3, 0.1]))    # current
    q_g = normalize(np.array([0.9, -0.1, 0.2, 0.3]))   # goal

    # Compute the error quaternion manually for reference
    q_e = quat_mult(quat_inv(q_g), q)
    if q_e[0] < 0:
        q_e = -q_e

    # quaternion_vector (default, already tested)
    law_qv = LawInterface(attitude_representation='quaternion_vector')
    out_qv = attitude_full_to_full(q_g, q, law_qv)
    check(out_qv.shape == (3,), "quat_vector: shape (3,)")
    check(norm(out_qv - q_e[1:4]) < 1e-12, "quat_vector: matches q_e[1:4]")

    # quaternion_full
    law_qf = LawInterface(attitude_representation='quaternion_full')
    out_qf = attitude_full_to_full(q_g, q, law_qf)
    check(out_qf.shape == (4,), "quat_full: shape (4,)")
    check(norm(out_qf - q_e) < 1e-12, "quat_full: matches full q_e")

    # MRP
    law_mrp = LawInterface(attitude_representation='mrp')
    out_mrp = attitude_full_to_full(q_g, q, law_mrp)
    expected_mrp = quat_to_mrp(q_e)
    check(out_mrp.shape == (3,), "mrp: shape (3,)")
    check(norm(out_mrp - expected_mrp) < 1e-12, "mrp: matches quat_to_mrp(q_e)")

    # Cayley
    law_cay = LawInterface(attitude_representation='cayley')
    out_cay = attitude_full_to_full(q_g, q, law_cay)
    expected_cay = quat_to_cayley(q_e)
    check(out_cay.shape == (3,), "cayley: shape (3,)")
    check(norm(out_cay - expected_cay) < 1e-12, "cayley: matches quat_to_cayley(q_e)")

    # DCM
    law_dcm = LawInterface(attitude_representation='dcm')
    out_dcm = attitude_full_to_full(q_g, q, law_dcm)
    expected_dcm = rot_mat(q_e)
    check(out_dcm.shape == (3, 3), "dcm: shape (3,3)")
    check(norm(out_dcm - expected_dcm) < 1e-12, "dcm: matches rot_mat(q_e)")

    # Euler 3-2-1
    law_euler = LawInterface(attitude_representation='euler_321')
    out_euler = attitude_full_to_full(q_g, q, law_euler)
    expected_euler = quat_to_euler(q_e)
    check(out_euler.shape == (3,), "euler_321: shape (3,)")
    check(norm(out_euler - expected_euler) < 1e-12, "euler_321: matches quat_to_euler(q_e)")

    # 2x MRP
    law_2mrp = LawInterface(attitude_representation='2mrp')
    out_2mrp = attitude_full_to_full(q_g, q, law_2mrp)
    expected_2mrp = 2.0 * quat_to_mrp(q_e)
    check(out_2mrp.shape == (3,), "2mrp: shape (3,)")
    check(norm(out_2mrp - expected_2mrp) < 1e-12, "2mrp: matches 2*quat_to_mrp(q_e)")

    # Zero error: all representations should give identity/zero
    for rep in ['quaternion_vector', 'mrp', 'cayley', '2mrp', 'euler_321']:
        law_z = LawInterface(attitude_representation=rep)
        z = attitude_full_to_full(q, q, law_z)
        check(norm(z) < 1e-10, f"zero error ({rep}): norm < 1e-10")

    law_zf = LawInterface(attitude_representation='quaternion_full')
    zf = attitude_full_to_full(q, q, law_zf)
    check(abs(zf[0] - 1.0) < 1e-10 and norm(zf[1:]) < 1e-10,
          "zero error (quat_full): identity quaternion")

    law_zd = LawInterface(attitude_representation='dcm')
    zd = attitude_full_to_full(q, q, law_zd)
    check(norm(zd - np.eye(3)) < 1e-10, "zero error (dcm): identity matrix")

    # zero_attitude helper
    check(norm(zero_attitude('quaternion_vector')) < 1e-12, "zero_attitude quat_vector")
    check(abs(zero_attitude('quaternion_full')[0] - 1.0) < 1e-12, "zero_attitude quat_full")
    check(norm(zero_attitude('mrp')) < 1e-12, "zero_attitude mrp")
    check(norm(zero_attitude('dcm') - np.eye(3)) < 1e-12, "zero_attitude dcm")


# ==================================================================
# Main
# ==================================================================
def main():
    print("=" * 60)
    print("Phase 2: Goal Formulation Unit Tests")
    print("=" * 60)

    test_find_perpendicular()
    test_quat_set_basis()
    test_nearest_quaternion()
    test_conventions()
    test_attitude_full_to_full()
    test_attitude_reduced_to_full()
    test_attitude_reduced_to_reduced()
    test_projection_matrix()
    test_omega_ref_analytical()
    test_full_goal_formulation_step()
    test_attitude_representations()

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
