r"""
Tests for lunisolar (Sun + Moon) third-body perturbations added to the orbit
dynamics.

The reference physics is the third-body disturbing potential

.. math::
    U_3 = \mu_3\left(\frac{1}{\lVert\mathbf{R}-\mathbf{s}\rVert}
          - \frac{\mathbf{R}\cdot\mathbf{s}}{\lVert\mathbf{s}\rVert^3}\right),

whose gradient is the perturbing acceleration. Because ``U_3`` carries a large
constant offset, a real finite difference cancels catastrophically, so the
gradient is taken by **complex-step** differentiation (machine precision) of an
independently written potential. The acceleration is additionally checked
against the analytic tidal expansion in the small-``R`` limit.
"""

import numpy as np
import pytest

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import (
    Orbital_State,
    _third_body_accel,
    _third_body_accel_jac,
)
from ADCS.orbits.universal_constants import ThirdBodyConstants

MU_SUN = ThirdBodyConstants.mu_sun
MU_MOON = ThirdBodyConstants.mu_moon
EPHEM = Ephemeris()

# Representative inertial body positions [km] (Sun ~1 AU, Moon ~3.84e5 km).
SUN = np.array([1.47098e8, 2.0e7, -1.1e7])
MOON = np.array([-3.1e5, 1.9e5, 8.0e4])
SUN_BODY = (MU_SUN, SUN)
MOON_BODY = (MU_MOON, MOON)


# --------------------------------------------------------------------------- #
# Independent reference physics
# --------------------------------------------------------------------------- #
def third_body_potential(R, bodies):
    # complex-safe: no np.linalg.norm / abs
    R = np.asarray(R)
    U = 0.0
    for mu, s in bodies:
        s = np.asarray(s, dtype=R.dtype)
        d = R - s
        rms = (s @ s) ** 1.5
        U = U + mu * (1.0 / np.sqrt(d @ d) - (R @ s) / rms)
    return U


def accel_from_potential_cs(R, bodies, h=1e-30):
    R0 = np.asarray(R, dtype=complex)
    g = np.zeros(3)
    for j in range(3):
        Rp = R0.copy()
        Rp[j] += 1j * h
        g[j] = np.imag(third_body_potential(Rp, bodies)) / h
    return g


def tidal_accel(R, bodies):
    out = np.zeros(3)
    for mu, s in bodies:
        s = np.asarray(s, dtype=float)
        sn = np.linalg.norm(s)
        sh = s / sn
        out += mu / sn**3 * (3.0 * (R @ sh) * sh - R)
    return out


def sample_positions(n=10, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        out.append(direction * rng.uniform(6700.0, 9500.0))
    return out


# --------------------------------------------------------------------------- #
# Acceleration equals the gradient of the third-body potential
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bodies", [[SUN_BODY], [MOON_BODY], [SUN_BODY, MOON_BODY]])
@pytest.mark.parametrize("R", sample_positions(6))
def test_third_body_accel_matches_potential_gradient(bodies, R):
    a_impl = _third_body_accel(R, bodies)
    a_ref = accel_from_potential_cs(R, bodies)
    assert np.allclose(a_impl, a_ref, rtol=1e-9, atol=1e-18)


# --------------------------------------------------------------------------- #
# Analytic Jacobian matches finite differences of the acceleration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("R", sample_positions(6))
def test_third_body_jacobian_matches_finite_difference(R):
    bodies = [SUN_BODY, MOON_BODY]
    jac = _third_body_accel_jac(R, bodies)
    fd = np.zeros((3, 3))
    h = 1.0
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        fd[:, j] = (_third_body_accel(R + e, bodies) - _third_body_accel(R - e, bodies)) / (2.0 * h)
    assert np.allclose(jac, fd, rtol=1e-6, atol=1e-18)


# --------------------------------------------------------------------------- #
# Physical signatures
# --------------------------------------------------------------------------- #
def test_third_body_vanishes_at_earth_center():
    # At R = 0 the direct and indirect terms cancel exactly.
    assert np.array_equal(_third_body_accel(np.zeros(3), [SUN_BODY, MOON_BODY]), np.zeros(3))


def test_third_body_is_linear_superposition():
    R = np.array([6900.0, 1500.0, -1200.0])
    a_both = _third_body_accel(R, [SUN_BODY, MOON_BODY])
    a_sum = _third_body_accel(R, [SUN_BODY]) + _third_body_accel(R, [MOON_BODY])
    assert np.allclose(a_both, a_sum, rtol=0.0, atol=1e-18)


@pytest.mark.parametrize("scale", [1.0, 1e-1, 1e-2, 1e-3])
def test_third_body_reduces_to_tidal_field_for_small_radius(scale):
    # The leading tidal term is exact as R/|s| -> 0; the relative error must
    # shrink with the radius (quadratically).
    R = scale * np.array([1.2, -0.7, 0.9])
    a = _third_body_accel(R, [SUN_BODY, MOON_BODY])
    a_tidal = tidal_accel(R, [SUN_BODY, MOON_BODY])
    rel = np.linalg.norm(a - a_tidal) / np.linalg.norm(a_tidal)
    assert rel < 5e-3 * scale + 1e-9


def test_moon_perturbation_exceeds_sun_at_leo():
    # The Moon raises a larger tide than the Sun at Earth (~2x), a classic check.
    state = Orbital_State(ephem=EPHEM, J2000=0.05, R=np.array([7000.0, 0.0, 1500.0]),
                          V=np.array([0.0, 7.4, 0.6]))
    bodies = state._lunisolar_bodies(True)
    a_sun = np.linalg.norm(_third_body_accel(state.R, [bodies[0]]))
    a_moon = np.linalg.norm(_third_body_accel(state.R, [bodies[1]]))
    assert a_moon > a_sun
    # Both are micro-scale accelerations (~1e-9 km/s^2) at LEO.
    assert 1e-11 < a_sun < 1e-8
    assert 1e-11 < a_moon < 1e-8


def test_get_moon_eci_distance_is_lunar():
    state = Orbital_State(ephem=EPHEM, J2000=0.0, R=np.array([7000.0, 0.0, 0.0]),
                          V=np.array([0.0, 7.5, 0.0]))
    moon = state.get_moon_eci()
    assert 3.5e5 < np.linalg.norm(moon) < 4.1e5  # 356,500-406,700 km perigee/apogee band


# --------------------------------------------------------------------------- #
# Backward compatibility and end-to-end plumbing
# --------------------------------------------------------------------------- #
def test_lunisolar_off_is_bit_identical_default():
    state = Orbital_State(ephem=EPHEM, J2000=0.0, R=np.array([7000.0, 0.0, 1500.0]),
                          V=np.array([0.0, 7.4, 0.6]))
    default = state.propagate_orbit_rk4(60.0)
    explicit_off = state.propagate_orbit_rk4(60.0, lunisolar=False)
    assert np.array_equal(default.R, explicit_off.R)
    assert np.array_equal(default.V, explicit_off.V)


def test_lunisolar_changes_propagated_trajectory():
    R = np.array([6878.0, 0.0, 0.0])
    V = np.array([0.0, 7.0, 2.8])
    s_off = Orbital_State(ephem=EPHEM, J2000=0.0, R=R, V=V)
    s_on = s_off.copy()
    period = 2.0 * np.pi * np.sqrt(np.linalg.norm(R) ** 3 / Orbital_State(
        ephem=EPHEM, J2000=0.0, R=R, V=V).mu_e)
    steps = 300
    dt = period / steps
    for _ in range(steps):
        s_off = s_off.propagate_orbit_rk4(dt, lunisolar=False)
        s_on = s_on.propagate_orbit_rk4(dt, lunisolar=True)
    drift = np.linalg.norm(s_on.R - s_off.R)
    assert drift > 1e-6   # third-body forces accumulate to a measurable offset
    assert drift < 10.0


def test_stm_rk4_with_lunisolar_and_zonals_matches_finite_difference():
    R0 = np.array([7000.0, 500.0, 1200.0])
    V0 = np.array([0.5, 7.2, 1.1])
    dt = 60.0
    base = Orbital_State(ephem=EPHEM, J2000=0.0, R=R0, V=V0)

    dr_dr0, dr_dv0, dv_dr0, dv_dv0 = base.propagate_jacobians_rk4(
        dt, zonal_order=6, lunisolar=True
    )
    stm = np.block([[dr_dr0, dr_dv0], [dv_dr0, dv_dv0]])

    def step(R, V):
        s = Orbital_State(ephem=EPHEM, J2000=0.0, R=R, V=V)
        out = s.propagate_orbit_rk4(dt, zonal_order=6, lunisolar=True)
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

    assert np.allclose(stm, fd, rtol=1e-4, atol=1e-6)


def test_orbit_batch_propagation_with_lunisolar_differs_from_without():
    from ADCS.orbits.orbit import Orbit
    from ADCS.orbits.universal_constants import TimeConstants

    os0 = Orbital_State(ephem=EPHEM, J2000=0.0, R=np.array([6878.0, 0.0, 0.0]),
                        V=np.array([0.0, 7.0, 2.8]))
    end = os0.J2000 + TimeConstants.sec2cent * 5400.0  # ~ one LEO orbit
    orb_off = Orbit(os0=os0, end_time=end, dt=30.0, verbose=False)
    orb_on = Orbit(os0=os0, end_time=end, dt=30.0, verbose=False, lunisolar=True)

    t_end = orb_off.max_time()
    r_off = orb_off.get_os(t_end).R
    r_on = orb_on.get_os(t_end).R
    assert np.linalg.norm(r_on - r_off) > 1e-6
