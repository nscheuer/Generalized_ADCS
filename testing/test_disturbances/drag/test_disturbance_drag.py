import numpy as np

from ADCS.satellite_hardware.disturbances import Drag_Disturbance, GeometryFace
from testing.test_disturbances._helpers import (
    fd_quat_hess,
    fd_quat_jac,
    make_geometry_config,
    make_orbital_state,
    make_satellite,
    make_state,
)


def expected_drag_torque(drag: Drag_Disturbance, sat, x: np.ndarray, os) -> np.ndarray:
    vecs = os.get_state_vector(x=x)
    velocity = vecs["v"] * 1000.0
    rho = vecs["rho"]
    face_factors = np.maximum(0.0, drag.normals @ velocity) * drag.areas * drag.CDs
    lever_sum = face_factors @ (drag.centroids - sat.COM)
    return -0.5 * rho * np.cross(lever_sum, velocity)


def test_drag_normalizes_face_normals():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.0, 0.2, 0.0]), normal=np.array([0.0, 5.0, 0.0]), CD=1.4)
        )
    )

    assert np.allclose(drag.normals[0], np.array([0.0, 1.0, 0.0]))


def test_drag_zero_density_produces_zero_torque():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.0, 0.2, 0.0]), normal=np.array([0.0, 1.0, 0.0]), CD=1.4)
        )
    )
    sat = make_satellite(COM=np.zeros(3), disturbances=[drag])
    x = make_state()
    os = make_orbital_state(V=np.array([0.2, 7.5, -0.1]), rho=0.0)

    torque = drag.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, np.zeros(3))


def test_drag_back_facing_face_contributes_zero():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.0, 0.3, 0.0]), normal=np.array([0.0, -1.0, 0.0]), CD=1.4)
        )
    )
    sat = make_satellite(COM=np.zeros(3), disturbances=[drag])
    x = make_state()
    os = make_orbital_state(V=np.array([0.1, 7.5, 0.2]), rho=3.0e-12)

    torque = drag.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, np.zeros(3))


def test_drag_single_face_matches_closed_form():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.5, centroid=np.array([0.1, 0.3, -0.2]), normal=np.array([0.0, 1.0, 0.0]), CD=1.7)
        )
    )
    sat = make_satellite(COM=np.array([0.02, -0.01, 0.04]), disturbances=[drag])
    x = make_state()
    os = make_orbital_state(V=np.array([0.2, 7.4, -0.1]), rho=4.0e-12)
    expected = expected_drag_torque(drag, sat, x, os)

    torque = drag.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, expected)


def test_drag_symmetric_faces_cancel():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.0, 0.2, 0.0]), normal=np.array([1.0, 0.0, 0.0]), CD=1.3),
            GeometryFace(area=2.0, centroid=np.array([0.0, -0.2, 0.0]), normal=np.array([1.0, 0.0, 0.0]), CD=1.3),
        )
    )
    sat = make_satellite(COM=np.zeros(3), disturbances=[drag])
    x = make_state()
    os = make_orbital_state(V=np.array([7.4, 0.0, 0.0]), rho=4.0e-12)

    torque = drag.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, np.zeros(3), atol=1.0e-15)


def test_drag_torque_qjac_matches_finite_difference():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.0, 0.3, 0.0]), normal=np.array([0.0, 2.0, 0.0]), CD=1.4)
        )
    )
    drag.active = 1.0
    sat = make_satellite(COM=np.zeros(3), disturbances=[drag])
    x = make_state(q=np.array([0.6, 0.2, -0.1, 0.4]))
    os = make_orbital_state(V=np.array([0.01, 7.5, 0.2]), rho=3.0e-12)
    expected = fd_quat_jac(lambda xx: drag.torque(sat=sat, x=xx, os=os), x)

    jacobian = drag.torque_qjac(sat=sat, x=x, os=os)

    assert np.allclose(jacobian, expected, atol=1.0e-4, rtol=2.0e-3)


def test_drag_torque_qqhess_matches_finite_difference():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.0, 0.3, 0.0]), normal=np.array([0.0, 2.0, 0.0]), CD=1.4)
        )
    )
    drag.active = 1.0
    sat = make_satellite(COM=np.zeros(3), disturbances=[drag])
    x = make_state(q=np.array([0.6, 0.2, -0.1, 0.4]))
    os = make_orbital_state(V=np.array([0.01, 7.5, 0.2]), rho=3.0e-12)
    expected = fd_quat_hess(lambda xx: drag.torque(sat=sat, x=xx, os=os), x)

    hessian = drag.torque_qqhess(sat=sat, x=x, os=os)

    assert np.allclose(hessian, expected, atol=4.0e-4, rtol=4.0e-3)


def test_drag_disturbance_is_used_in_satellite_dynamics():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.5, centroid=np.array([0.1, 0.3, -0.2]), normal=np.array([0.0, 1.0, 0.0]), CD=1.7)
        )
    )
    sat = make_satellite(COM=np.array([0.02, -0.01, 0.04]), disturbances=[drag])
    x = make_state()
    os = make_orbital_state(V=np.array([0.2, 7.4, -0.1]), rho=4.0e-12)
    torque = drag.torque(sat=sat, x=x, os=os)

    xdot = sat.dynamics_core(x=x, u=np.zeros(0), orbital_state=os)

    assert np.allclose(xdot[:3], sat.invJ_noRW @ torque)
    assert np.allclose(xdot[3:7], np.zeros(4))
