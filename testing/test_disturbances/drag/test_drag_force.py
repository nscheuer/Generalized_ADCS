"""Aerodynamic drag force (the torque's companion).

The force is the per-face flat-plate drag summed without lever arms, so the
torque must equal the sum of (centroid - COM) x (per-face force).  That
consistency check ties the new force to the already-tested torque.
"""

import numpy as np

from ADCS.satellite_hardware.disturbances import Drag_Disturbance, GeometryFace
from testing.test_disturbances._helpers import (
    fd_quat_hess_raw,
    fd_quat_jac_raw,
    make_geometry_config,
    make_orbital_state,
    make_satellite,
    make_state,
)


def per_face_drag_forces(drag: Drag_Disturbance, x, os) -> np.ndarray:
    vecs = os.get_state_vector(x=x)
    velocity = vecs["v"] * 1000.0
    rho = vecs["rho"]
    face_factors = np.maximum(0.0, drag.normals @ velocity) * drag.areas * drag.CDs
    return -0.5 * rho * face_factors[:, None] * velocity[None, :]


def test_drag_force_single_face_matches_closed_form():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.5, centroid=np.array([0.1, 0.3, -0.2]), normal=np.array([0.0, 1.0, 0.0]), CD=1.7)
        )
    )
    sat = make_satellite(disturbances=[drag])
    x = make_state()
    os = make_orbital_state(V=np.array([0.0, 7.4, 0.0]), rho=4.0e-12)

    force = drag.force(sat=sat, x=x, os=os)

    speed = 7400.0
    expected = -0.5 * 4.0e-12 * 1.7 * 2.5 * speed**2 * np.array([0.0, 1.0, 0.0])
    assert np.allclose(force, expected)
    assert force @ np.array([0.0, 1.0, 0.0]) < 0.0  # opposes the velocity


def test_drag_force_zero_density_is_zero():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.zeros(3), normal=np.array([0.0, 1.0, 0.0]), CD=1.4)
        )
    )
    sat = make_satellite(disturbances=[drag])
    os = make_orbital_state(V=np.array([0.2, 7.5, -0.1]), rho=0.0)

    assert np.allclose(drag.force(sat=sat, x=make_state(), os=os), np.zeros(3))


def test_drag_force_back_facing_face_contributes_zero():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.zeros(3), normal=np.array([0.0, -1.0, 0.0]), CD=1.4)
        )
    )
    sat = make_satellite(disturbances=[drag])
    os = make_orbital_state(V=np.array([0.1, 7.5, 0.2]), rho=3.0e-12)

    assert np.allclose(drag.force(sat=sat, x=make_state(), os=os), np.zeros(3))


def test_drag_force_opposing_faces_only_count_the_ram_face():
    ram = GeometryFace(area=2.0, centroid=np.array([0.0, 0.2, 0.0]), normal=np.array([0.0, 1.0, 0.0]), CD=2.2)
    wake = GeometryFace(area=2.0, centroid=np.array([0.0, -0.2, 0.0]), normal=np.array([0.0, -1.0, 0.0]), CD=2.2)
    both = Drag_Disturbance(make_geometry_config(ram, wake))
    ram_only = Drag_Disturbance(make_geometry_config(ram))
    sat = make_satellite(disturbances=[both])
    x = make_state(q=np.array([0.9, 0.1, -0.2, 0.3]))
    os = make_orbital_state(V=np.array([0.3, 7.4, -0.5]), rho=4.0e-12)

    assert np.allclose(both.force(sat=sat, x=x, os=os), ram_only.force(sat=sat, x=x, os=os))


def test_drag_torque_is_lever_arm_cross_per_face_force():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.5, centroid=np.array([0.1, 0.3, -0.2]), normal=np.array([0.0, 1.0, 0.0]), CD=1.7),
            GeometryFace(area=1.0, centroid=np.array([-0.3, 0.0, 0.1]), normal=np.array([1.0, 1.0, 0.0]), CD=2.0),
        )
    )
    sat = make_satellite(COM=np.array([0.02, -0.01, 0.04]), disturbances=[drag])
    x = make_state(q=np.array([0.7, -0.2, 0.3, 0.1]))
    os = make_orbital_state(V=np.array([0.2, 7.4, -0.1]), rho=4.0e-12)

    per_face = per_face_drag_forces(drag, x, os)
    expected_torque = np.sum(np.cross(drag.centroids - sat.COM, per_face), axis=0)

    assert np.allclose(drag.force(sat=sat, x=x, os=os), per_face.sum(axis=0))
    assert np.allclose(drag.torque(sat=sat, x=x, os=os), expected_torque)


def test_drag_force_qjac_matches_finite_difference():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.0, 0.3, 0.0]), normal=np.array([0.0, 2.0, 0.0]), CD=1.4),
            GeometryFace(area=0.5, centroid=np.array([0.2, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.5]), CD=2.0),
        )
    )
    sat = make_satellite(disturbances=[drag])
    x = make_state(q=np.array([0.6, 0.2, -0.1, 0.4]))
    os = make_orbital_state(V=np.array([0.01, 7.5, 0.2]), rho=3.0e-12)
    expected = fd_quat_jac_raw(lambda xx: drag.force(sat=sat, x=xx, os=os), x)

    jacobian = drag.force_qjac(sat=sat, x=x, os=os)

    assert jacobian.shape == (4, 3)
    assert np.allclose(jacobian, expected, atol=1.0e-9, rtol=1.0e-6)


def test_drag_force_qqhess_matches_finite_difference():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.0, 0.3, 0.0]), normal=np.array([0.0, 2.0, 0.0]), CD=1.4),
            GeometryFace(area=0.5, centroid=np.array([0.2, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.5]), CD=2.0),
        )
    )
    sat = make_satellite(disturbances=[drag])
    x = make_state(q=np.array([0.6, 0.2, -0.1, 0.4]))
    os = make_orbital_state(V=np.array([0.01, 7.5, 0.2]), rho=3.0e-12)
    expected = fd_quat_hess_raw(lambda xx: drag.force(sat=sat, x=xx, os=os), x)

    hessian = drag.force_qqhess(sat=sat, x=x, os=os)

    assert hessian.shape == (4, 4, 3)
    assert np.allclose(hessian, expected, atol=1.0e-7, rtol=1.0e-5)


def test_drag_force_reaches_satellite_dist_forces():
    drag = Drag_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.5, centroid=np.array([0.1, 0.3, -0.2]), normal=np.array([0.0, 1.0, 0.0]), CD=1.7)
        )
    )
    sat = make_satellite(disturbances=[drag])
    x = make_state()
    os = make_orbital_state(V=np.array([0.2, 7.4, -0.1]), rho=4.0e-12)

    assert np.allclose(sat.dist_forces(x=x, os=os), drag.force(sat=sat, x=x, os=os))
