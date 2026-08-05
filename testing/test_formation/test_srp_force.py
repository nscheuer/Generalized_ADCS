r"""
SRP FORCE gates. The force path shares its per-face optics with the existing
torque path, so the tests that matter are the ones proving that sharing is
real -- not that the arithmetic runs.
"""
import numpy as np
import pytest

from ADCS.satellite_hardware.disturbances.helpers.geometry_config import (
    GeometryConfig, GeometryFace)
from ADCS.satellite_hardware.disturbances.srp_disturbance import (
    srp_force_body, _srp_force_kernel, _srp_torque_kernel)
from ADCS.orbits.universal_constants import EarthConstants

S0, CL = EarthConstants.solar_constant, EarthConstants.c


def _box(eta_s=0.5, eta_d=0.2, eta_a=0.3, a=0.03):
    """A 6-face box with +/- pairs, the factory convention."""
    faces = []
    for n in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]):
        n = np.array(n, dtype=float)
        faces.append({"area": a, "centroid": 0.1 * n, "normal": n,
                      "eta_s": eta_s, "eta_d": eta_d, "eta_a": eta_a, "CD": 2.2})
    return GeometryConfig([GeometryFace.from_dict(f) for f in faces])


def test_force_is_the_sum_the_torque_kernel_already_computes():
    r"""THE point of this addition: the torque kernel computes per-face forces
    internally and discards them. Rebuild the torque from the force pieces and
    require it to match the torque kernel exactly, which proves the two paths
    share one optical model rather than merely agreeing today."""
    cfg = _box()
    p = cfg.params
    normals = np.array([f["normal"] for f in p], dtype=float)
    areas = np.array([f["area"] for f in p], dtype=float)
    cents = np.array([f["centroid"] for f in p], dtype=float)
    es = np.array([f["eta_s"] for f in p], dtype=float)
    ed = np.array([f["eta_d"] for f in p], dtype=float)
    ea = np.array([f["eta_a"] for f in p], dtype=float)
    S_B = np.array([1.4e8, 3.0e7, -2.0e7])
    R_B = np.array([7.0e3, -1.0e3, 2.0e3])
    COM = np.array([0.05, -0.02, 0.03])   # OFF-centre, so the torque is
                                          # genuinely non-zero and the r x F
                                          # assembly is actually exercised

    # per-face force, reconstructed from the SAME multipliers
    sb = (S_B - R_B) / np.linalg.norm(S_B - R_B)
    F_sum = np.zeros(3)
    T_sum = np.zeros(3)
    for i in range(len(p)):
        cg = max(0.0, float(normals[i] @ sb))
        A_cg = areas[i] * cg
        ms = A_cg * (ea[i] + ed[i])
        mn = A_cg * (2.0 * es[i] * cg + (2.0 / 3.0) * ed[i])
        Fi = -(S0 / CL) * (ms * sb + mn * normals[i])
        F_sum += Fi
        T_sum += np.cross(cents[i] - COM, Fi)

    F = _srp_force_kernel(S_B, R_B, normals, areas, es, ed, ea, S0, CL)
    T = _srp_torque_kernel(S_B, R_B, COM, normals, areas, cents, es, ed, ea, S0, CL)
    assert np.allclose(F, F_sum, rtol=1e-12, atol=0.0), (F, F_sum)
    assert np.linalg.norm(T_sum) > 0.0, "pick a COM that yields a non-zero torque"
    assert np.allclose(T, T_sum, rtol=1e-12, atol=0.0), (T, T_sum)


def test_force_points_away_from_the_sun():
    cfg = _box()
    S_B = np.array([1.5e8, 0.0, 0.0])
    R_B = np.zeros(3)
    F = srp_force_body(S_B, R_B, cfg)
    sb = (S_B - R_B) / np.linalg.norm(S_B - R_B)
    assert float(F @ sb) < 0.0                       # pushed away from the Sun
    assert abs(F[1]) < 1e-18 and abs(F[2]) < 1e-18   # symmetric box, no lateral net


def test_pure_absorber_magnitude_is_the_textbook_value():
    r"""eta_a = 1: a fully absorbing plate normal to the Sun feels exactly
    P = S0/c times its area, directed along the photon travel direction."""
    A = 2.5
    face = [{"area": A, "centroid": np.zeros(3), "normal": np.array([1.0, 0, 0]),
             "eta_s": 0.0, "eta_d": 0.0, "eta_a": 1.0, "CD": 0.0}]
    cfg = GeometryConfig([GeometryFace.from_dict(face[0])])
    F = srp_force_body(np.array([1.5e8, 0, 0]), np.zeros(3), cfg)
    assert np.isclose(np.linalg.norm(F), (S0 / CL) * A, rtol=1e-12)
    assert F[0] < 0.0


def test_specular_mirror_is_twice_the_absorber_at_normal_incidence():
    A = 2.5
    def cfg_for(es, ea):
        return GeometryConfig([GeometryFace.from_dict({"area": A, "centroid": np.zeros(3),
                                "normal": np.array([1.0, 0, 0]), "eta_s": es,
                                "eta_d": 0.0, "eta_a": ea, "CD": 0.0})])
    S_B, R_B = np.array([1.5e8, 0, 0]), np.zeros(3)
    F_abs = srp_force_body(S_B, R_B, cfg_for(0.0, 1.0))
    F_mir = srp_force_body(S_B, R_B, cfg_for(1.0, 0.0))
    assert np.isclose(np.linalg.norm(F_mir), 2.0 * np.linalg.norm(F_abs), rtol=1e-12)


def test_back_faces_contribute_nothing():
    r"""Faces are one-sided (cos^+ clipping), matching the torque path and the
    +/- pair convention the geometry factories emit."""
    A = 1.0
    front = GeometryConfig([GeometryFace.from_dict({"area": A, "centroid": np.zeros(3),
                             "normal": np.array([1.0, 0, 0]), "eta_s": 0.0,
                             "eta_d": 0.0, "eta_a": 1.0, "CD": 0.0})])
    back = GeometryConfig([GeometryFace.from_dict({"area": A, "centroid": np.zeros(3),
                            "normal": np.array([-1.0, 0, 0]), "eta_s": 0.0,
                            "eta_d": 0.0, "eta_a": 1.0, "CD": 0.0})])
    S_B, R_B = np.array([1.5e8, 0, 0]), np.zeros(3)
    assert np.linalg.norm(srp_force_body(S_B, R_B, front)) > 0.0
    assert np.allclose(srp_force_body(S_B, R_B, back), 0.0)


@pytest.mark.parametrize("ang_deg", [0.0, 30.0, 60.0, 85.0])
def test_absorber_follows_the_cosine_law(ang_deg):
    A = 3.0
    cfg = GeometryConfig([GeometryFace.from_dict({"area": A, "centroid": np.zeros(3),
                           "normal": np.array([1.0, 0, 0]), "eta_s": 0.0,
                           "eta_d": 0.0, "eta_a": 1.0, "CD": 0.0})])
    th = np.radians(ang_deg)
    S_B = 1.5e8 * np.array([np.cos(th), np.sin(th), 0.0])
    F = srp_force_body(S_B, np.zeros(3), cfg)
    assert np.isclose(np.linalg.norm(F), (S0 / CL) * A * np.cos(th), rtol=1e-10)
