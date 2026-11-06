"""Sentman free-molecular drag at finite speed ratio.

Checks the cold-beam limit against Drag_Disturbance, the edge-on thermal
floor, its dependence on gas temperature, torque/force consistency and the
attitude derivatives.
"""

import numpy as np

from ADCS.satellite_hardware.disturbances import (
    Drag_Disturbance,
    GeometryFace,
    Sentman_Drag_Disturbance,
)
from testing.test_disturbances._helpers import (
    fd_quat_hess_raw,
    fd_quat_jac_raw,
    make_geometry_config,
    make_orbital_state,
    make_satellite,
    make_state,
)

RHO = 3.0e-12
V_ECI = np.array([0.0, 7.5, 0.0])  # km/s, along body +y at identity attitude
YHAT = np.array([0.0, 1.0, 0.0])


def cube(half: float = 0.05, area: float = 0.01, CD: float = 2.0):
    faces = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            n = np.zeros(3)
            n[axis] = sign
            faces.append(GeometryFace(area=area, centroid=half * n, normal=n, CD=CD))
    return make_geometry_config(*faces)


def cold_beam(config):
    """Sentman with S -> infinity and no re-emission: the cold-beam flat plate."""
    return Sentman_Drag_Disturbance(config, T_inf=1.0e-6, T_wall=0.0, alpha=1.0)


def test_cold_beam_limit_matches_drag_disturbance_with_cd_two():
    config = make_geometry_config(
        GeometryFace(area=0.02, centroid=np.array([0.0, 0.3, 0.0]), normal=np.array([0.0, 1.0, 0.0]), CD=2.0),
        GeometryFace(area=0.01, centroid=np.array([0.2, 0.0, 0.1]), normal=np.array([1.0, 0.5, -0.2]), CD=2.0),
        GeometryFace(area=0.03, centroid=np.array([0.0, -0.3, 0.0]), normal=np.array([0.0, -1.0, 0.0]), CD=2.0),
    )
    sentman = cold_beam(config)
    classic = Drag_Disturbance(config)
    sat = make_satellite(COM=np.array([0.01, -0.02, 0.03]), disturbances=[sentman])
    x = make_state(q=np.array([0.9, 0.1, -0.2, 0.3]))
    os = make_orbital_state(V=np.array([0.3, 7.4, -0.5]), rho=RHO)

    assert np.allclose(sentman.force(sat=sat, x=x, os=os), classic.force(sat=sat, x=x, os=os), rtol=1.0e-9)
    assert np.allclose(sentman.torque(sat=sat, x=x, os=os), classic.torque(sat=sat, x=x, os=os), rtol=1.0e-9)


def test_edge_on_pair_has_thermal_floor_that_cold_beam_lacks():
    pair = make_geometry_config(
        GeometryFace(area=0.5, centroid=np.zeros(3), normal=np.array([0.0, 0.0, 1.0])),
        GeometryFace(area=0.5, centroid=np.zeros(3), normal=np.array([0.0, 0.0, -1.0])),
    )
    sentman = Sentman_Drag_Disturbance(pair, T_inf=900.0)
    classic = Drag_Disturbance(pair)
    sat = make_satellite(disturbances=[sentman])
    x = make_state()
    os = make_orbital_state(V=V_ECI, rho=RHO)

    f_cold = classic.force(sat=sat, x=x, os=os)
    f_sentman = sentman.force(sat=sat, x=x, os=os)

    assert np.allclose(f_cold, np.zeros(3))
    speed = 7500.0
    S = sentman.speed_ratio(speed)
    expected_drag = 2.0 * 0.5 * RHO * speed**2 * 0.5 * (1.0 / (S * np.sqrt(np.pi)))
    assert np.isclose(-f_sentman @ YHAT, expected_drag, rtol=1.0e-9)
    assert np.allclose(f_sentman - (f_sentman @ YHAT) * YHAT, np.zeros(3), atol=1.0e-20)  # lift cancels


def test_hotter_gas_raises_the_floor():
    pair = make_geometry_config(
        GeometryFace(area=0.5, centroid=np.zeros(3), normal=np.array([0.0, 0.0, 1.0])),
        GeometryFace(area=0.5, centroid=np.zeros(3), normal=np.array([0.0, 0.0, -1.0])),
    )
    sat = make_satellite()
    x = make_state()
    os = make_orbital_state(V=V_ECI, rho=RHO)
    cold = Sentman_Drag_Disturbance(pair, T_inf=600.0).force(sat=sat, x=x, os=os)
    hot = Sentman_Drag_Disturbance(pair, T_inf=1500.0).force(sat=sat, x=x, os=os)

    assert np.isclose((-hot @ YHAT) / (-cold @ YHAT), np.sqrt(1500.0 / 600.0), rtol=1.0e-9)


def test_back_face_is_shadowed_at_orbital_speed_ratio():
    back = make_geometry_config(GeometryFace(area=1.0, centroid=np.zeros(3), normal=-YHAT))
    ram = make_geometry_config(GeometryFace(area=1.0, centroid=np.zeros(3), normal=YHAT))
    sat = make_satellite()
    x = make_state()
    os = make_orbital_state(V=V_ECI, rho=RHO)
    f_back = Sentman_Drag_Disturbance(back).force(sat=sat, x=x, os=os)
    f_ram = Sentman_Drag_Disturbance(ram).force(sat=sat, x=x, os=os)

    assert np.linalg.norm(f_back) < 1.0e-12 * np.linalg.norm(f_ram)
    assert f_ram @ YHAT < 0.0


def test_ram_face_drag_coefficient_is_sentman_closed_form():
    ram = make_geometry_config(GeometryFace(area=1.0, centroid=np.zeros(3), normal=YHAT))
    sentman = Sentman_Drag_Disturbance(ram, T_inf=900.0, T_wall=300.0, alpha=0.93)
    speed = 7500.0
    C_D, C_L = sentman.face_coefficients(speed * YHAT)
    S = sentman.speed_ratio(speed)
    # gamma = 1: Z = 2, P = exp(-S^2)/S ~ 1e-28, so C_D = 2 (1 + 1/(2 S^2)) + v_re/V (sqrt(pi) + P/2).
    expected = 2.0 + 1.0 / S**2 + np.sqrt(np.pi) * sentman.reemission_ratio(speed)

    assert np.isclose(C_D[0], expected, rtol=1.0e-9)
    assert S > 5.0  # orbital speed ratio: the finite-S correction is ~1/S^2 ~ 2 percent
    assert np.isclose(C_L[0], 0.0, atol=1.0e-12)


def test_zero_density_gives_zero_force_torque_and_derivatives():
    sentman = Sentman_Drag_Disturbance(cube())
    sat = make_satellite(disturbances=[sentman])
    x = make_state(q=np.array([0.9, 0.1, -0.2, 0.3]))
    os = make_orbital_state(V=V_ECI, rho=0.0)

    assert np.array_equal(sentman.force(sat=sat, x=x, os=os), np.zeros(3))
    assert np.array_equal(sentman.torque(sat=sat, x=x, os=os), np.zeros(3))
    assert np.array_equal(sentman.force_qjac(sat=sat, x=x, os=os), np.zeros((4, 3)))
    assert np.array_equal(sentman.torque_qqhess(sat=sat, x=x, os=os), np.zeros((4, 4, 3)))


def test_centred_cube_has_no_torque_and_offset_com_gives_minus_com_cross_force():
    sentman = Sentman_Drag_Disturbance(cube())
    x = make_state(q=np.array([0.9, 0.1, -0.2, 0.3]))
    os = make_orbital_state(V=np.array([0.3, 7.4, -0.5]), rho=RHO)
    centred = make_satellite(COM=np.zeros(3), disturbances=[sentman])
    offset_com = np.array([0.01, -0.02, 0.03])
    offset = make_satellite(COM=offset_com, disturbances=[sentman])

    force = sentman.force(sat=centred, x=x, os=os)

    assert np.allclose(sentman.torque(sat=centred, x=x, os=os), np.zeros(3), atol=1.0e-18)
    assert np.allclose(sentman.torque(sat=offset, x=x, os=os), -np.cross(offset_com, force))


def test_torque_is_lever_arm_cross_face_forces():
    sentman = Sentman_Drag_Disturbance(cube())
    sat = make_satellite(COM=np.array([0.01, -0.02, 0.03]), disturbances=[sentman])
    x = make_state(q=np.array([0.9, 0.1, -0.2, 0.3]))
    os = make_orbital_state(V=np.array([0.3, 7.4, -0.5]), rho=RHO)
    V_B = os.get_state_vector(x=x)["v"] * 1000.0
    faces = sentman.face_forces(V_B, RHO)

    assert np.allclose(sentman.force(sat=sat, x=x, os=os), faces.sum(axis=0))
    assert np.allclose(sentman.torque(sat=sat, x=x, os=os), np.sum(np.cross(sentman.centroids - sat.COM, faces), axis=0))


def test_attitude_jacobians_match_finite_difference():
    sentman = Sentman_Drag_Disturbance(cube())
    sat = make_satellite(COM=np.array([0.01, -0.02, 0.03]), disturbances=[sentman])
    x = make_state(q=np.array([0.6, 0.2, -0.1, 0.4]))
    os = make_orbital_state(V=np.array([0.01, 7.5, 0.2]), rho=RHO)

    f_jac = sentman.force_qjac(sat=sat, x=x, os=os)
    t_jac = sentman.torque_qjac(sat=sat, x=x, os=os)
    f_expected = fd_quat_jac_raw(lambda xx: sentman.force(sat=sat, x=xx, os=os), x)
    t_expected = fd_quat_jac_raw(lambda xx: sentman.torque(sat=sat, x=xx, os=os), x)

    assert np.allclose(f_jac, f_expected, rtol=1.0e-5, atol=1.0e-9 * np.abs(f_expected).max())
    assert np.allclose(t_jac, t_expected, rtol=1.0e-5, atol=1.0e-9 * np.abs(t_expected).max())


def test_attitude_hessians_match_finite_difference():
    sentman = Sentman_Drag_Disturbance(cube())
    sat = make_satellite(COM=np.array([0.01, -0.02, 0.03]), disturbances=[sentman])
    x = make_state(q=np.array([0.6, 0.2, -0.1, 0.4]))
    os = make_orbital_state(V=np.array([0.01, 7.5, 0.2]), rho=RHO)

    f_hess = sentman.force_qqhess(sat=sat, x=x, os=os)
    t_hess = sentman.torque_qqhess(sat=sat, x=x, os=os)
    f_expected = fd_quat_hess_raw(lambda xx: sentman.force(sat=sat, x=xx, os=os), x)
    t_expected = fd_quat_hess_raw(lambda xx: sentman.torque(sat=sat, x=xx, os=os), x)

    assert np.allclose(f_hess, f_expected, rtol=1.0e-4, atol=1.0e-6 * np.abs(f_expected).max())
    assert np.allclose(t_hess, t_expected, rtol=1.0e-4, atol=1.0e-6 * np.abs(t_expected).max())


def test_sentman_drag_reaches_satellite_sums_and_dynamics():
    sentman = Sentman_Drag_Disturbance(cube())
    sat = make_satellite(COM=np.array([0.01, -0.02, 0.03]), disturbances=[sentman])
    x = make_state(q=np.array([0.9, 0.1, -0.2, 0.3]))
    os = make_orbital_state(V=np.array([0.3, 7.4, -0.5]), rho=RHO)
    torque = sentman.torque(sat=sat, x=x, os=os)

    assert np.allclose(sat.dist_forces(x=x, os=os), sentman.force(sat=sat, x=x, os=os))
    assert np.allclose(sat.dist_torques(x=x, os=os), torque)
    xdot = sat.dynamics_core(x=x, u=np.zeros(0), orbital_state=os)
    assert np.allclose(xdot[:3], sat.invJ_noRW @ torque)
