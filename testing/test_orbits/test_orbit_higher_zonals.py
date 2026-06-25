r"""
Tests for the higher-order zonal gravity harmonics (J3-J6) added on top of the
existing J2 orbit dynamics.

The reference physics is built *independently* of the implementation under test:
the perturbing geopotential

.. math::
    U_{pert} = -\frac{\mu}{r} \sum_{n} J_n \left(\frac{R_e}{r}\right)^n P_n(z/r)

is evaluated with :func:`numpy.polynomial.legendre.legval` (a different code
path from the recurrence used in ``orbital_state``) and finite-differenced to
obtain the acceleration. The implementation must reproduce that gradient, the
classical J2 closed form, the per-degree parity of zonal harmonics, and a
position-Jacobian / state-transition matrix consistent with finite differences.
"""

import numpy as np
import pytest
from numpy.polynomial import legendre as L

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import (
    Orbital_State,
    _zonal_perturbation_accel,
    _zonal_perturbation_accel_jac,
)
from ADCS.orbits.universal_constants import EarthConstants

MU = EarthConstants.mu_e
RE = EarthConstants.R_e
JC = EarthConstants.Jcoeffs  # [J2, J3, J4, J5, J6]
EPHEM = Ephemeris()


# --------------------------------------------------------------------------- #
# Independent reference physics
# --------------------------------------------------------------------------- #
def perturbing_potential(R, coeffs, start_degree):
    R = np.asarray(R, dtype=float)
    r = np.linalg.norm(R)
    s = R[2] / r
    U = 0.0
    for k, Jn in enumerate(coeffs):
        n = start_degree + k
        basis = np.zeros(n + 1)
        basis[n] = 1.0
        Pn = L.legval(s, basis)
        U += -(MU / r) * Jn * (RE / r) ** n * Pn
    return U


def accel_from_potential(R, coeffs, start_degree, h=1.0):
    # a = grad(U): the code uses a = +grad(U) (two-body grad(mu/r) = -mu R/r^3).
    g = np.zeros(3)
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        g[j] = (
            perturbing_potential(R + e, coeffs, start_degree)
            - perturbing_potential(R - e, coeffs, start_degree)
        ) / (2.0 * h)
    return g


def sample_positions(n=12, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        out.append(direction * rng.uniform(6700.0, 9500.0))
    return out


# --------------------------------------------------------------------------- #
# Degree-2 consistency with the legacy J2 closed form
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("R", sample_positions(6))
def test_general_zonal_reproduces_legacy_j2_exactly(R):
    V = np.zeros(3)
    _, vdot_j2 = Orbital_State._orbit_dynamics_raw(R, V, MU, RE, JC[0], True)
    _, vdot_2body = Orbital_State._orbit_dynamics_raw(R, V, MU, RE, JC[0], False)
    a_legacy = vdot_j2 - vdot_2body
    a_general = _zonal_perturbation_accel(R, MU, RE, [JC[0]], start_degree=2)
    assert np.allclose(a_general, a_legacy, rtol=0.0, atol=1e-18)


# --------------------------------------------------------------------------- #
# Each harmonic's acceleration equals the gradient of its potential
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", [3, 4, 5, 6])
@pytest.mark.parametrize("R", sample_positions(8))
def test_single_harmonic_accel_matches_potential_gradient(degree, R):
    coeff = [JC[degree - 2]]
    a_impl = _zonal_perturbation_accel(R, MU, RE, coeff, start_degree=degree)
    a_ref = accel_from_potential(R, coeff, start_degree=degree)
    assert np.allclose(a_impl, a_ref, rtol=1e-5, atol=1e-20)


@pytest.mark.parametrize("R", sample_positions(8))
def test_combined_j3_to_j6_accel_matches_potential_gradient(R):
    a_impl = _zonal_perturbation_accel(R, MU, RE, JC[1:], start_degree=3)
    a_ref = accel_from_potential(R, JC[1:], start_degree=3)
    assert np.allclose(a_impl, a_ref, rtol=1e-5, atol=1e-20)


# --------------------------------------------------------------------------- #
# Parity: zonal harmonic of degree n has potential parity (-1)^n in z, so the
# acceleration of an even harmonic flips only its z-component under z -> -z,
# while an odd harmonic flips only its in-plane (x, y) components. This is the
# defining signature that the *odd* terms (J3, J5) are present and correct.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", [2, 3, 4, 5, 6])
def test_zonal_harmonic_has_correct_z_parity(degree):
    R = np.array([6900.0, -1500.0, 2200.0])
    R_mirror = np.array([R[0], R[1], -R[2]])
    coeff = [JC[degree - 2]]
    a = _zonal_perturbation_accel(R, MU, RE, coeff, start_degree=degree)
    a_m = _zonal_perturbation_accel(R_mirror, MU, RE, coeff, start_degree=degree)
    if degree % 2 == 0:  # even harmonic: in-plane same, z flips
        assert np.allclose(a_m[:2], a[:2], rtol=1e-12, atol=1e-20)
        assert np.allclose(a_m[2], -a[2], rtol=1e-12, atol=1e-20)
    else:                # odd harmonic: in-plane flips, z same
        assert np.allclose(a_m[:2], -a[:2], rtol=1e-12, atol=1e-20)
        assert np.allclose(a_m[2], a[2], rtol=1e-12, atol=1e-20)


# --------------------------------------------------------------------------- #
# Magnitude hierarchy: at LEO every higher harmonic is far smaller than J2, and
# the combined J3-J6 contribution is a small (<1%) correction to J2.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("R", sample_positions(6))
def test_higher_harmonics_are_small_corrections_to_j2(R):
    a_j2 = _zonal_perturbation_accel(R, MU, RE, [JC[0]], start_degree=2)
    a_high = _zonal_perturbation_accel(R, MU, RE, JC[1:], start_degree=3)
    assert np.linalg.norm(a_high) < 0.01 * np.linalg.norm(a_j2)


# --------------------------------------------------------------------------- #
# Backward compatibility: the new plumbing leaves J2-only behavior bit-identical.
# --------------------------------------------------------------------------- #
def test_higher_zonals_none_is_bit_identical_to_legacy_call():
    R = np.array([7000.0, 1200.0, -800.0])
    V = np.array([1.0, 7.0, 0.5])
    a_legacy = Orbital_State._orbit_dynamics_raw(R, V, MU, RE, JC[0], True)[1]
    a_new = Orbital_State._orbit_dynamics_raw(R, V, MU, RE, JC[0], True, higher_zonals=None)[1]
    assert np.array_equal(a_legacy, a_new)


def test_propagate_default_equals_zonal_order_two():
    state = Orbital_State(ephem=EPHEM, J2000=0.1, R=np.array([7000.0, 0.0, 1500.0]),
                          V=np.array([0.0, 7.4, 0.6]))
    default = state.propagate_orbit_rk4(45.0)
    explicit = state.propagate_orbit_rk4(45.0, zonal_order=2)
    assert np.array_equal(default.R, explicit.R)
    assert np.array_equal(default.V, explicit.V)


def test_empty_higher_zonals_contributes_nothing():
    R = np.array([7200.0, 300.0, -1100.0])
    assert np.array_equal(
        _zonal_perturbation_accel(R, MU, RE, [], start_degree=3), np.zeros(3)
    )
    assert np.array_equal(
        _zonal_perturbation_accel_jac(R, MU, RE, [], start_degree=3), np.zeros((3, 3))
    )


# --------------------------------------------------------------------------- #
# Enabling higher harmonics measurably changes a propagated trajectory.
# --------------------------------------------------------------------------- #
def test_zonal_order_six_changes_trajectory_vs_j2_only():
    R = np.array([6878.0, 0.0, 0.0])
    V = np.array([0.0, 7.0, 2.8])  # inclined so odd/even zonals both act
    s2 = Orbital_State(ephem=EPHEM, J2000=0.0, R=R, V=V)
    s6 = s2.copy()
    period = 2.0 * np.pi * np.sqrt(np.linalg.norm(R) ** 3 / MU)
    steps = 400
    dt = period / steps
    for _ in range(steps):
        s2 = s2.propagate_orbit_rk4(dt, zonal_order=2)
        s6 = s6.propagate_orbit_rk4(dt, zonal_order=6)
    drift = np.linalg.norm(s6.R - s2.R)
    # J3-J6 are tiny, but over a full orbit they accumulate to a clearly
    # non-zero, sub-kilometre-to-kilometre separation.
    assert drift > 1e-4
    assert drift < 50.0


# --------------------------------------------------------------------------- #
# Jacobians: the full position-Jacobian (analytic J2 + complex-step J3-J6) and
# the RK4 state-transition matrix must match finite differences of the dynamics.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("R", sample_positions(6))
def test_dynamics_jacobian_with_higher_zonals_matches_finite_difference(R):
    V = np.array([0.0, 7.0, 1.0])

    def accel(Rv):
        return Orbital_State._orbit_dynamics_raw(
            Rv, V, MU, RE, JC[0], True, higher_zonals=JC[1:]
        )[1]

    _, _, dvd_dr, _ = Orbital_State._orbit_dynamics_jacobians_raw(
        R, MU, RE, JC[0], True, higher_zonals=JC[1:]
    )
    fd = np.zeros((3, 3))
    h = 1e-2
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        fd[:, j] = (accel(R + e) - accel(R - e)) / (2.0 * h)
    assert np.allclose(dvd_dr, fd, rtol=1e-6, atol=1e-14)


def test_stm_rk4_with_higher_zonals_matches_finite_difference():
    R0 = np.array([7000.0, 500.0, 1200.0])
    V0 = np.array([0.5, 7.2, 1.1])
    dt = 60.0
    base = Orbital_State(ephem=EPHEM, J2000=0.0, R=R0, V=V0)

    dr_dr0, dr_dv0, dv_dr0, dv_dv0 = base.propagate_jacobians_rk4(dt, zonal_order=6)
    stm = np.block([[dr_dr0, dr_dv0], [dv_dr0, dv_dv0]])

    def step(R, V):
        s = Orbital_State(ephem=EPHEM, J2000=0.0, R=R, V=V)
        out = s.propagate_orbit_rk4(dt, zonal_order=6)
        return np.concatenate([out.R, out.V])

    fd = np.zeros((6, 6))
    hs = [1e-2, 1e-2, 1e-2, 1e-5, 1e-5, 1e-5]
    x0 = np.concatenate([R0, V0])
    for j in range(6):
        e = np.zeros(6)
        e[j] = hs[j]
        xp = x0 + e
        xm = x0 - e
        fd[:, j] = (step(xp[:3], xp[3:]) - step(xm[:3], xm[3:])) / (2.0 * hs[j])

    # Tolerance set by the O(h^2) truncation of the central-difference STM on the
    # small gravity-gradient coupling blocks; the dominant (dt-coupling) terms
    # agree far more tightly.
    assert np.allclose(stm, fd, rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------- #
# zonal_order is clamped to the available coefficients and disabled with J2 off.
# --------------------------------------------------------------------------- #
def test_zonal_order_above_six_is_clamped():
    state = Orbital_State(ephem=EPHEM, J2000=0.0, R=np.array([7000.0, 0.0, 1000.0]),
                          V=np.array([0.0, 7.4, 0.3]))
    a6 = state.orbit_dynamics(zonal_order=6)[1]
    a99 = state.orbit_dynamics(zonal_order=99)[1]
    assert np.array_equal(a6, a99)


def test_j2_off_disables_all_zonals():
    state = Orbital_State(ephem=EPHEM, J2000=0.0, R=np.array([7000.0, 0.0, 1000.0]),
                          V=np.array([0.0, 7.4, 0.3]))
    a_off = state.orbit_dynamics(J2_perturbation_on=False, zonal_order=6)[1]
    a_two_body = -MU * state.R / np.linalg.norm(state.R) ** 3
    assert np.allclose(a_off, a_two_body, rtol=0.0, atol=1e-18)
