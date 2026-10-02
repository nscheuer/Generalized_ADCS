import numpy as np

from ADCS.orbits.universal_constants import EarthConstants
from ADCS.satellite_hardware.disturbances import GeometryFace, SRP_Disturbance
from testing.test_disturbances._helpers import (
    fd_quat_hess,
    fd_quat_jac,
    make_geometry_config,
    make_orbital_state,
    make_satellite,
    make_state,
    resolve_method,
)


def expected_srp_torque(srp: SRP_Disturbance, sat, x: np.ndarray, os) -> np.ndarray:
    if not os.is_sunlit():
        return np.zeros(3)
    vecs = os.get_state_vector(x=x)
    sun_direction = vecs["s"] - vecs["r"]
    sun_direction = sun_direction / np.linalg.norm(sun_direction)
    torque = np.zeros(3)
    for area, centroid, normal, eta_s, eta_d, eta_a in zip(
        srp.areas,
        srp.centroids,
        srp.normals,
        srp.eta_s,
        srp.eta_d,
        srp.eta_a,
    ):
        cosine = max(0.0, float(normal @ sun_direction))
        projected_area = area * cosine
        lever = centroid - sat.COM
        tangential = projected_area * (eta_a + eta_d) * np.cross(lever, sun_direction)
        normal_term = projected_area * (2.0 * eta_s * cosine + (2.0 / 3.0) * eta_d) * np.cross(lever, normal)
        torque += tangential + normal_term
    return -(EarthConstants.solar_constant / EarthConstants.c) * torque


def test_srp_normalizes_face_normals():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=1.5, centroid=np.array([0.0, 0.2, 0.0]), normal=np.array([5.0, 0.0, 0.0]), eta_a=1.0)
        )
    )

    assert np.allclose(srp.normals[0], np.array([1.0, 0.0, 0.0]))


def test_srp_eclipse_produces_zero_torque():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=1.5, centroid=np.array([0.0, 0.2, 0.0]), normal=np.array([1.0, 0.0, 0.0]), eta_a=1.0)
        )
    )
    sat = make_satellite(COM=np.zeros(3), disturbances=[srp])
    x = make_state()
    os = make_orbital_state(
        R=np.array([7000.0, 0.0, 0.0]),
        S=np.array([1.0e8 + 7000.0, 0.0, 0.0]),
        sunlit=False,
    )

    torque = srp.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, np.zeros(3))


def test_srp_back_facing_face_contributes_zero():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=1.5, centroid=np.array([0.0, 0.2, 0.0]), normal=np.array([-1.0, 0.0, 0.0]), eta_a=1.0)
        )
    )
    sat = make_satellite(COM=np.zeros(3), disturbances=[srp])
    x = make_state()
    os = make_orbital_state(
        R=np.array([7000.0, 0.0, 0.0]),
        S=np.array([1.0e8 + 7000.0, 0.0, 0.0]),
        sunlit=True,
    )

    torque = srp.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, np.zeros(3))


def test_srp_absorptive_face_matches_closed_form():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=1.5, centroid=np.array([0.0, 0.2, 0.0]), normal=np.array([1.0, 0.0, 0.0]), eta_a=1.0)
        )
    )
    sat = make_satellite(COM=np.zeros(3), disturbances=[srp])
    x = make_state()
    os = make_orbital_state(
        R=np.array([7000.0, 0.0, 0.0]),
        S=np.array([1.0e8 + 7000.0, 0.0, 0.0]),
        sunlit=True,
    )
    expected = expected_srp_torque(srp, sat, x, os)

    torque = srp.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, expected)


def test_srp_mixed_optical_coefficients_match_closed_form():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(
                area=2.0,
                centroid=np.array([0.2, 0.1, -0.3]),
                normal=np.array([1.0, 0.0, 0.0]),
                eta_s=0.2,
                eta_d=0.3,
                eta_a=0.5,
            )
        )
    )
    sat = make_satellite(COM=np.array([0.01, -0.02, 0.03]), disturbances=[srp])
    x = make_state()
    os = make_orbital_state(
        R=np.array([7000.0, 0.0, 0.0]),
        S=np.array([1.0e8 + 7000.0, 0.0, 0.0]),
        sunlit=True,
    )
    expected = expected_srp_torque(srp, sat, x, os)

    torque = srp.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, expected)


def test_srp_symmetric_faces_cancel():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=1.5, centroid=np.array([0.0, 0.2, 0.0]), normal=np.array([1.0, 0.0, 0.0]), eta_s=0.2, eta_d=0.3, eta_a=0.5),
            GeometryFace(area=1.5, centroid=np.array([0.0, -0.2, 0.0]), normal=np.array([1.0, 0.0, 0.0]), eta_s=0.2, eta_d=0.3, eta_a=0.5),
        )
    )
    sat = make_satellite(COM=np.zeros(3), disturbances=[srp])
    x = make_state()
    os = make_orbital_state(
        R=np.array([7000.0, 0.0, 0.0]),
        S=np.array([1.0e8 + 7000.0, 0.0, 0.0]),
        sunlit=True,
    )

    torque = srp.torque(sat=sat, x=x, os=os)

    assert np.allclose(torque, np.zeros(3), atol=1.0e-15)


def test_srp_torque_qjac_matches_finite_difference():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=1.5, centroid=np.array([0.2, 0.0, 0.1]), normal=np.array([0.0, 0.0, 5.0]), eta_s=0.2, eta_d=0.3, eta_a=0.5)
        )
    )
    sat = make_satellite(COM=np.zeros(3), disturbances=[srp])
    x = make_state(q=np.array([0.8, -0.1, 0.2, 0.3]))
    os = make_orbital_state(R=np.array([7000.0, 20.0, -10.0]), S=np.array([1.0e8, -2.0e8, 3.0e8]), sunlit=True)
    qjac = resolve_method(srp, "torque_qjac", "torque_qjav")
    expected = fd_quat_jac(lambda xx: srp.torque(sat=sat, x=xx, os=os), x)

    jacobian = qjac(sat=sat, x=x, os=os)

    assert np.allclose(jacobian, expected, atol=2.0e-5, rtol=2.0e-4)


def test_srp_torque_qqhess_matches_finite_difference():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=1.5, centroid=np.array([0.2, 0.0, 0.1]), normal=np.array([0.0, 0.0, 5.0]), eta_s=0.2, eta_d=0.3, eta_a=0.5)
        )
    )
    sat = make_satellite(COM=np.zeros(3), disturbances=[srp])
    x = make_state(q=np.array([0.8, -0.1, 0.2, 0.3]))
    os = make_orbital_state(R=np.array([7000.0, 20.0, -10.0]), S=np.array([1.0e8, -2.0e8, 3.0e8]), sunlit=True)
    expected = fd_quat_hess(lambda xx: srp.torque(sat=sat, x=xx, os=os), x)

    hessian = srp.torque_qqhess(sat=sat, x=x, os=os)

    assert np.allclose(hessian, expected, atol=2.0e-5, rtol=2.0e-4)


def test_srp_disturbance_is_used_in_satellite_dynamics():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=1.5, centroid=np.array([0.0, 0.2, 0.0]), normal=np.array([1.0, 0.0, 0.0]), eta_a=1.0)
        )
    )
    sat = make_satellite(COM=np.zeros(3), disturbances=[srp])
    x = make_state()
    os = make_orbital_state(
        R=np.array([7000.0, 0.0, 0.0]),
        S=np.array([1.0e8 + 7000.0, 0.0, 0.0]),
        sunlit=True,
    )
    torque = srp.torque(sat=sat, x=x, os=os)

    xdot = sat.dynamics_core(x=x, u=np.zeros(0), orbital_state=os)

    assert np.allclose(xdot[:3], sat.invJ_noRW @ torque)
    assert np.allclose(xdot[3:7], np.zeros(4))
