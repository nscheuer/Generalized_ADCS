import numpy as np

from ADCS.orbits.universal_constants import EarthConstants
from ADCS.satellite_hardware.disturbances import GG_Disturbance
from testing.test_disturbances._helpers import make_orbital_state, make_satellite, make_state


def expected_gg_torque(sat, x: np.ndarray, os) -> np.ndarray:
    radius = os.get_state_vector(x=x)["r"]
    nadir = -radius / np.linalg.norm(radius)
    return 3.0 * EarthConstants.mu_e / (np.linalg.norm(radius) ** 3) * np.cross(nadir, sat.J_0 @ nadir)


def test_gg_spherical_inertia_produces_zero_torque():
    sat = make_satellite(J_0=4.0 * np.eye(3))
    gg = GG_Disturbance()
    x = make_state()
    os = make_orbital_state(R=np.array([7000.0, 20.0, -10.0]))

    torque = gg.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, np.zeros(3))


def test_gg_principal_axis_alignment_produces_zero_torque():
    sat = make_satellite(J_0=np.diag([2.0, 3.0, 5.0]))
    gg = GG_Disturbance()
    x = make_state()
    os = make_orbital_state(R=np.array([7000.0, 0.0, 0.0]))

    torque = gg.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, np.zeros(3))


def test_gg_torque_matches_closed_form():
    sat = make_satellite(J_0=np.diag([2.0, 3.0, 5.0]))
    gg = GG_Disturbance()
    x = make_state()
    os = make_orbital_state(R=np.array([7000.0, 500.0, -300.0]))
    expected = expected_gg_torque(sat, x, os)

    torque = gg.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, expected)


def test_gg_torque_scales_with_inverse_cube_radius():
    sat = make_satellite(J_0=np.diag([2.0, 3.0, 5.0]))
    gg = GG_Disturbance()
    x = make_state()
    os_near = make_orbital_state(R=np.array([7000.0, 500.0, -300.0]))
    os_far = make_orbital_state(R=2.0 * np.array([7000.0, 500.0, -300.0]))
    torque_near = gg.torque(sat=sat, x=x, os=os_near)
    torque_far = gg.torque(sat=sat, x=x, os=os_far)

    assert np.allclose(torque_far, torque_near / 8.0)


def test_gg_torque_uses_body_frame_position_from_attitude():
    sat = make_satellite(J_0=np.diag([2.0, 3.0, 5.0]))
    gg = GG_Disturbance()
    x = make_state(q=np.array([0.6, 0.2, -0.1, 0.4]))
    os = make_orbital_state(R=np.array([7000.0, 500.0, -300.0]))
    expected = expected_gg_torque(sat, x, os)

    torque = gg.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, expected)


def test_gg_satellite_dist_torques_matches_single_disturbance():
    gg = GG_Disturbance()
    sat = make_satellite(J_0=np.diag([2.0, 3.0, 5.0]), disturbances=[gg])
    x = make_state()
    os = make_orbital_state(R=np.array([7000.0, 500.0, -300.0]))

    total = sat.dist_torques(x=x, os=os)
    single = gg.torque(sat=sat, x=x, os=os)

    assert np.allclose(total, single)


def test_gg_disturbance_is_used_in_satellite_dynamics():
    gg = GG_Disturbance()
    sat = make_satellite(J_0=np.diag([2.0, 3.0, 5.0]), disturbances=[gg])
    x = make_state()
    os = make_orbital_state(R=np.array([7000.0, 500.0, -300.0]))
    torque = gg.torque(sat=sat, x=x, os=os)

    xdot = sat.dynamics_core(x=x, u=np.zeros(0), orbital_state=os)

    assert np.allclose(xdot[:3], sat.invJ_noRW @ torque)
    assert np.allclose(xdot[3:7], np.zeros(4))
