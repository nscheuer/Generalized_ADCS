import numpy as np

from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.satellite_hardware.satellite import Satellite

from testing.test_satellite._helpers import expected_quat_dot, make_mtqs, make_orbital_state, make_rws


def test_dynamics_core_matches_plain_quaternion_kinematics_at_identity():
    sat = Satellite()
    os = make_orbital_state()
    x = np.hstack([np.array([0.01, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])])

    dx = sat.dynamics_core(x=x, u=np.array([]), orbital_state=os)

    assert np.allclose(dx, np.array([0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0]))


def test_dynamics_core_matches_plain_quaternion_kinematics_for_rotated_quaternion():
    sat = Satellite()
    os = make_orbital_state()
    w = np.array([0.01, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0, 0.0])

    dx = sat.dynamics_core(x=np.hstack([w, q]), u=np.array([]), orbital_state=os)

    assert np.allclose(dx[3:], expected_quat_dot(w, q))
    assert np.allclose(dx[:3], np.zeros(3))


def test_dynamics_core_matches_closed_form_torque_free_rigid_body_case():
    qJ = normalize(np.array([0.5, -0.3, 0.1, 0.8]))
    J_0 = np.diagflat([2.0, 3.0, 10.0])
    J_body = rot_mat(qJ) @ J_0 @ rot_mat(qJ).T

    q0 = normalize(np.array([0.4, -0.2, 0.7, 0.5]))
    R = rot_mat(q0)
    J_eci = R @ J_body @ R.T
    w0 = 0.05 * normalize(np.array([0.2, -0.4, 0.7]))
    w_eci = R @ w0
    H_eci = J_eci @ w_eci

    expected_wdot = -R.T @ np.linalg.inv(J_eci) @ np.cross(w_eci, H_eci)
    expected_qdot = expected_quat_dot(w0, q0)

    sat = Satellite(J_0=J_body)
    dx = sat.dynamics_core(x=np.concatenate([w0, q0]), u=np.array([]), orbital_state=make_orbital_state())

    assert np.allclose(dx, np.concatenate([expected_wdot, expected_qdot]))


def test_mtq_dynamics_match_axis_aligned_cases():
    os = make_orbital_state(B=np.array([1.0e-5, 0.0, 0.0]))
    x = np.hstack([np.array([0.01, 0.0, 0.0]), MathConstants.zeroquat])
    mtqs = make_mtqs()

    sat_x = Satellite(actuators=[mtqs[0]])
    sat_y = Satellite(actuators=[mtqs[1]])
    sat_z = Satellite(actuators=[mtqs[2]])

    assert np.allclose(sat_x.dynamics_core(x=x, u=np.array([1.0]), orbital_state=os), np.array([0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0]))
    assert np.allclose(sat_y.dynamics_core(x=x, u=np.array([1.0]), orbital_state=os), np.array([0.0, 0.0, -1.0e-5, 0.0, 0.005, 0.0, 0.0]))
    assert np.allclose(sat_z.dynamics_core(x=x, u=np.array([1.0]), orbital_state=os), np.array([0.0, 1.0e-5, 0.0, 0.0, 0.005, 0.0, 0.0]))


def test_mtq_dynamics_match_superposed_multi_axis_inputs():
    os = make_orbital_state(B=np.array([1.0e-5, 0.0, 0.0]))
    x = np.hstack([np.array([0.01, 0.0, 0.0]), MathConstants.zeroquat])
    sat = Satellite(actuators=make_mtqs())

    assert np.allclose(sat.dynamics_core(x=x, u=np.array([1.0, 0.0, 0.0]), orbital_state=os), np.array([0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0]))
    assert np.allclose(sat.dynamics_core(x=x, u=np.array([0.0, 1.0, 0.0]), orbital_state=os), np.array([0.0, 0.0, -1.0e-5, 0.0, 0.005, 0.0, 0.0]))
    assert np.allclose(sat.dynamics_core(x=x, u=np.array([0.0, 0.0, 1.0]), orbital_state=os), np.array([0.0, 1.0e-5, 0.0, 0.0, 0.005, 0.0, 0.0]))


def test_reaction_wheel_dynamics_match_expected_storage_and_body_terms():
    sat = Satellite(actuators=make_rws())
    os = make_orbital_state()
    x = np.concatenate([0.01 * MathConstants.unitvecs[0], MathConstants.zeroquat, np.array([0.1, 0.0, 0.0])])
    u = np.array([0.021, -0.05, 0.0])

    dx = sat.dynamics_core(x=x, u=u, orbital_state=os)

    clipped_u0 = np.clip(u[0], -sat.actuators[0].u_max, sat.actuators[0].u_max)
    clipped_u1 = np.clip(u[1], -sat.actuators[1].u_max, sat.actuators[1].u_max)
    expected = np.array([
        clipped_u0 / (1.0 - 0.001),
        clipped_u1 / (1.0 - 0.002),
        0.0,
        0.0,
        0.005,
        0.0,
        0.0,
        -clipped_u0 / (1.0 - 0.001),
        abs(clipped_u1) / (1.0 - 0.002),
        0.0,
    ])
    assert np.allclose(dx, expected, rtol=1e-8, atol=1e-8)


def test_dist_torques_and_act_torque_are_zero_without_disturbances_or_actuators():
    sat = Satellite()
    os = make_orbital_state()
    x = np.hstack([np.zeros(3), MathConstants.zeroquat])

    assert np.allclose(sat.dist_torques(x=x, os=os), np.zeros(3))
    assert np.allclose(sat.act_torque(x=x, u=np.array([]), os=os), np.zeros(3))
