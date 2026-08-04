"""
Attitude-error covariance UNITS contract + the sanctioned angle-space tools.

Finding (verified, latent — no current consumer mis-uses it, but the hazard
is real and undocumented): an estimator attitude-error covariance block
``cov[3:6, 3:6]`` is in the ``vec_mode`` 3-parameter, NOT the physical
attitude angle. For ``vec_mode=6`` the small-angle gain is G=0.5, so the
raw block is wrong by ``G^-2 = 4`` if read as an angle covariance
(empirically measured NEES ratio earlier was 3.91 ≈ 4).

These tests pin: (a) the gain G against the true parametrisation, (b) the
conversion against the *true nonlinear* vec3->rotation-vector mapping (not
just the formula), (c) the robust accessor / 3-sigma / sampler tools, and
(d) the hazard itself (raw vs angle differ by exactly G^2).
"""

import numpy as np
import pytest

from ADCS.helpers.math_helpers import (
    quat_to_vec3, vec3_to_quat,
    vec_mode_angle_gain, attitude_cov_param_to_angle,
    attitude_angle_covariance, attitude_angle_sigma,
    sample_attitude_error_rotvec,
)

MRP_MODES = [0, 1, 5, 6]
ALL_MODES = [0, 1, 3, 4, 5, 6]


def _rotvec_of_quat(q):
    """Physical rotation vector (rad) of a scalar-first unit quaternion."""
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    vn = np.linalg.norm(q[1:])
    if vn < 1e-15:
        return np.zeros(3)
    ang = 2.0 * np.arctan2(vn, q[0])
    return ang * q[1:] / vn


@pytest.mark.parametrize("mode", ALL_MODES)
def test_vec_mode_angle_gain_matches_true_parametrisation(mode):
    G = vec_mode_angle_gain(mode)
    # Independent FD of |quat_to_vec3| per radian of true rotation.
    phi = 1e-6
    q = np.array([np.cos(phi / 2), np.sin(phi / 2), 0.0, 0.0])
    G_fd = np.linalg.norm(quat_to_vec3(q, mode)) / phi
    assert np.isclose(G, G_fd, rtol=1e-4), f"mode {mode}: G {G} vs FD {G_fd}"
    if mode in (0, 1):
        assert np.isclose(G, 0.25, rtol=1e-3)
    if mode in (5, 6):
        assert np.isclose(G, 0.5, rtol=1e-3)


def test_param_to_angle_scaling_and_shapes():
    G = vec_mode_angle_gain(6)
    # Full reduced cov: rate block + attitude block + a rate<->attitude cross.
    P = np.zeros((6, 6))
    P[0:3, 0:3] = np.diag([1e-4, 2e-4, 3e-4])      # rate (untouched)
    P[3:6, 3:6] = np.diag([0.01, 0.02, 0.03])      # attitude (vec_mode param)
    P[0:3, 3:6] = 1e-3 * np.eye(3)
    P[3:6, 0:3] = P[0:3, 3:6].T

    A = attitude_angle_covariance(P, 6)
    assert np.allclose(A[0:3, 0:3], P[0:3, 0:3])                 # rate unchanged
    assert np.allclose(A[3:6, 3:6], P[3:6, 3:6] / G ** 2)        # block / G^2
    assert np.allclose(A[0:3, 3:6], P[0:3, 3:6] / G)             # cross / G
    assert np.allclose(A, A.T)                                   # symmetric
    # (3,3)-only path consistent with the full path's attitude block.
    assert np.allclose(attitude_angle_covariance(P[3:6, 3:6], 6),
                       A[3:6, 3:6])


@pytest.mark.parametrize("mode", MRP_MODES)
def test_conversion_matches_true_nonlinear_mapping(mode):
    """The linear angle-covariance must match the empirical covariance of the
    EXACT nonlinear vec3 -> rotation-vector map near zero (validates 1/G^2)."""
    rng = np.random.default_rng(0)
    G = vec_mode_angle_gain(mode)
    sig = 2e-3 / G                      # small param std (=> ~2 mrad attitude)
    Cp = np.diag([sig ** 2, (0.7 * sig) ** 2, (1.3 * sig) ** 2])
    Lp = np.linalg.cholesky(Cp)
    N = 200000
    vp = (Lp @ rng.standard_normal((3, N))).T
    phis = np.array([_rotvec_of_quat(vec3_to_quat(v, mode)) for v in vp])
    emp = np.cov(phis.T)
    lin = attitude_cov_param_to_angle(Cp, mode)
    # Scale-aware: max abs error must be small vs the covariance magnitude.
    # Off-diagonal MC scatter is ~ diag/sqrt(N) ~ 1e-7 (negligible); a wrong
    # gain (e.g. missing 1/G^2 -> 4x) would blow the diagonal by ~3*scale.
    scale = np.max(np.abs(np.diag(lin)))
    err = np.max(np.abs(emp - lin))
    assert err < 0.05 * scale, \
        f"mode {mode}: max|emp-lin|={err:.2e} vs 0.05*scale={0.05*scale:.2e}" \
        f"\nempirical\n{emp}\nlinear\n{lin}"


def test_angle_sigma_robust():
    P = np.zeros((6, 6))
    P[3:6, 3:6] = np.diag([0.04, 0.09, 1e-18])     # last ~0
    G = vec_mode_angle_gain(6)
    s3 = attitude_angle_sigma(P, 6, k=3.0)
    assert np.allclose(s3[:2], 3.0 * np.sqrt(np.array([0.04, 0.09])) / G)
    assert np.isfinite(s3).all() and s3[2] >= 0.0
    # negative diagonal (finite-precision sqrt filter) must not crash / nan
    P[3, 3] = -1e-15
    assert np.isfinite(attitude_angle_sigma(P, 6)).all()


def test_sampler_reproduces_angle_covariance_and_is_psd_robust():
    rng = np.random.default_rng(1)
    P = np.zeros((6, 6))
    P[3:6, 3:6] = np.array([[0.02, 0.005, 0.0],
                            [0.005, 0.03, 0.0],
                            [0.0, 0.0, 1e-20]])    # PSD, one ~0 eigenvalue
    s = sample_attitude_error_rotvec(P, 6, rng=rng, size=400000)
    assert s.shape == (400000, 3)
    target = attitude_angle_covariance(P, 6)[3:6, 3:6]
    assert np.allclose(np.cov(s.T), target, rtol=0.05, atol=1e-6)


def test_units_hazard_is_exactly_G_squared():
    """Encodes the finding: reading the raw vec_mode block as an angle
    covariance is wrong by exactly G^2 (=4 for vec_mode=6 ~ the 3.91
    empirically measured)."""
    P = np.eye(6)
    raw = P[3:6, 3:6]
    ang = attitude_angle_covariance(P, 6)[3:6, 3:6]
    G = vec_mode_angle_gain(6)
    assert np.allclose(ang, raw / G ** 2)
    assert np.isclose(1.0 / G ** 2, 4.0, rtol=1e-3)
