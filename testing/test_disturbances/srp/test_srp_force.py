"""Solar radiation pressure force (the torque's companion).

Per face the force is -(S0/c) [ m_s s_b + m_n n ] with m_s the absorbed-plus-
diffuse multiplier along the Sun line and m_n the specular-plus-diffuse
multiplier along the face normal.  The torque must be the sum of
(centroid - COM) x (per-face force), which ties the new force to the
already-tested torque.
"""

import numpy as np

from ADCS.orbits.universal_constants import EarthConstants
from ADCS.satellite_hardware.disturbances import GeometryFace, SRP_Disturbance
from testing.test_disturbances._helpers import (
    fd_quat_hess_raw,
    fd_quat_jac_raw,
    make_geometry_config,
    make_orbital_state,
    make_satellite,
    make_state,
)

PRESSURE = EarthConstants.solar_constant / EarthConstants.c


def sun_direction_body(x, os) -> np.ndarray:
    vecs = os.get_state_vector(x=x)
    look = vecs["s"] - vecs["r"]
    return look / np.linalg.norm(look)


def per_face_srp_forces(srp: SRP_Disturbance, x, os) -> np.ndarray:
    if not os.is_sunlit():
        return np.zeros((len(srp.areas), 3))
    s_body = sun_direction_body(x, os)
    cosine = np.maximum(0.0, srp.normals @ s_body)
    m_s = srp.areas * (srp.eta_a + srp.eta_d) * cosine
    m_n = srp.areas * cosine * (2.0 * srp.eta_s * cosine + (2.0 / 3.0) * srp.eta_d)
    return -PRESSURE * (m_s[:, None] * s_body[None, :] + m_n[:, None] * srp.normals)


def sunlit_os(**overrides):
    values = dict(R=np.array([7000.0, 0.0, 0.0]), S=np.array([1.0e8 + 7000.0, 0.0, 0.0]), sunlit=True)
    values.update(overrides)
    return make_orbital_state(**values)


def test_srp_force_eclipse_is_zero():
    srp = SRP_Disturbance(
        make_geometry_config(GeometryFace(area=1.5, centroid=np.zeros(3), normal=np.array([1.0, 0.0, 0.0]), eta_a=1.0))
    )
    sat = make_satellite(disturbances=[srp])
    x = make_state()
    os = sunlit_os(sunlit=False)

    assert np.allclose(srp.force(sat=sat, x=x, os=os), np.zeros(3))
    assert np.allclose(srp.force_qjac(sat=sat, x=x, os=os), np.zeros((4, 3)))
    assert np.allclose(srp.force_qqhess(sat=sat, x=x, os=os), np.zeros((4, 4, 3)))


def test_srp_force_absorptive_face_pushes_away_from_sun():
    srp = SRP_Disturbance(
        make_geometry_config(GeometryFace(area=1.5, centroid=np.zeros(3), normal=np.array([1.0, 0.0, 0.0]), eta_a=1.0))
    )
    sat = make_satellite(disturbances=[srp])
    x = make_state()
    os = sunlit_os()

    force = srp.force(sat=sat, x=x, os=os)

    # Sun along +x in the body frame, face normal +x: force is -P A along +x.
    assert np.allclose(force, -PRESSURE * 1.5 * np.array([1.0, 0.0, 0.0]))


def test_srp_force_specular_face_at_normal_incidence_is_twice_absorptive():
    face = dict(area=1.5, centroid=np.zeros(3), normal=np.array([1.0, 0.0, 0.0]))
    absorptive = SRP_Disturbance(make_geometry_config(GeometryFace(eta_a=1.0, **face)))
    specular = SRP_Disturbance(make_geometry_config(GeometryFace(eta_s=1.0, **face)))
    sat = make_satellite(disturbances=[absorptive, specular])
    x = make_state()
    os = sunlit_os()

    f_abs = absorptive.force(sat=sat, x=x, os=os)
    f_spec = specular.force(sat=sat, x=x, os=os)

    assert np.allclose(f_spec, 2.0 * f_abs)


def test_srp_force_back_facing_face_contributes_zero():
    srp = SRP_Disturbance(
        make_geometry_config(GeometryFace(area=1.5, centroid=np.zeros(3), normal=np.array([-1.0, 0.0, 0.0]), eta_a=1.0))
    )
    sat = make_satellite(disturbances=[srp])

    assert np.allclose(srp.force(sat=sat, x=make_state(), os=sunlit_os()), np.zeros(3))


def test_srp_force_mixed_coefficients_match_closed_form():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.2, 0.1, -0.3]), normal=np.array([1.0, 0.5, 0.0]), eta_s=0.2, eta_d=0.3, eta_a=0.5),
            GeometryFace(area=0.7, centroid=np.array([-0.1, 0.0, 0.2]), normal=np.array([0.0, 0.0, 1.0]), eta_s=0.6, eta_d=0.1, eta_a=0.3),
        )
    )
    sat = make_satellite(disturbances=[srp])
    x = make_state(q=np.array([0.8, 0.1, -0.3, 0.2]))
    os = sunlit_os()

    force = srp.force(sat=sat, x=x, os=os)

    assert np.allclose(force, per_face_srp_forces(srp, x, os).sum(axis=0))


def test_srp_torque_is_lever_arm_cross_per_face_force():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.2, 0.1, -0.3]), normal=np.array([1.0, 0.5, 0.0]), eta_s=0.2, eta_d=0.3, eta_a=0.5),
            GeometryFace(area=0.7, centroid=np.array([-0.1, 0.0, 0.2]), normal=np.array([0.0, 0.0, 1.0]), eta_s=0.6, eta_d=0.1, eta_a=0.3),
        )
    )
    sat = make_satellite(COM=np.array([0.01, -0.02, 0.03]), disturbances=[srp])
    x = make_state(q=np.array([0.8, 0.1, -0.3, 0.2]))
    os = sunlit_os()

    per_face = per_face_srp_forces(srp, x, os)
    expected_torque = np.sum(np.cross(srp.centroids - sat.COM, per_face), axis=0)

    assert np.allclose(srp.torque(sat=sat, x=x, os=os), expected_torque)


def test_srp_force_qjac_matches_finite_difference():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.2, 0.1, -0.3]), normal=np.array([1.0, 0.5, 0.0]), eta_s=0.2, eta_d=0.3, eta_a=0.5),
            GeometryFace(area=0.7, centroid=np.array([-0.1, 0.0, 0.2]), normal=np.array([0.0, 0.0, 1.0]), eta_s=0.6, eta_d=0.1, eta_a=0.3),
        )
    )
    sat = make_satellite(disturbances=[srp])
    x = make_state(q=np.array([0.8, 0.1, -0.3, 0.2]))
    os = sunlit_os()
    expected = fd_quat_jac_raw(lambda xx: srp.force(sat=sat, x=xx, os=os), x)

    jacobian = srp.force_qjac(sat=sat, x=x, os=os)

    assert jacobian.shape == (4, 3)
    assert np.allclose(jacobian, expected, atol=1.0e-12, rtol=1.0e-6)


def test_srp_force_qqhess_matches_finite_difference():
    srp = SRP_Disturbance(
        make_geometry_config(
            GeometryFace(area=2.0, centroid=np.array([0.2, 0.1, -0.3]), normal=np.array([1.0, 0.5, 0.0]), eta_s=0.2, eta_d=0.3, eta_a=0.5),
            GeometryFace(area=0.7, centroid=np.array([-0.1, 0.0, 0.2]), normal=np.array([0.0, 0.0, 1.0]), eta_s=0.6, eta_d=0.1, eta_a=0.3),
        )
    )
    sat = make_satellite(disturbances=[srp])
    x = make_state(q=np.array([0.8, 0.1, -0.3, 0.2]))
    os = sunlit_os()
    expected = fd_quat_hess_raw(lambda xx: srp.force(sat=sat, x=xx, os=os), x)

    hessian = srp.force_qqhess(sat=sat, x=x, os=os)

    assert hessian.shape == (4, 4, 3)
    assert np.allclose(hessian, expected, atol=1.0e-10, rtol=1.0e-5)


def test_srp_force_reaches_satellite_dist_forces():
    srp = SRP_Disturbance(
        make_geometry_config(GeometryFace(area=1.5, centroid=np.zeros(3), normal=np.array([1.0, 0.0, 0.0]), eta_a=1.0))
    )
    sat = make_satellite(disturbances=[srp])
    x = make_state()
    os = sunlit_os()

    assert np.allclose(sat.dist_forces(x=x, os=os), srp.force(sat=sat, x=x, os=os))
