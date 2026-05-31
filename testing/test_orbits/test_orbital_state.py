import numpy as np
import numdifftools as nd
import pytest

from ADCS.orbits.universal_constants import EarthConstants, TimeConstants

from testing.test_orbits._helpers import make_random_orbital_state, make_reference_orbital_state


def closest_approach(reference, trajectory, min_skip=1):
    d2 = np.sum((trajectory - reference) ** 2, axis=1)
    if min_skip > 0:
        d2[:min_skip] = np.inf
    i_star = np.argmin(d2)
    return i_star, np.sqrt(d2[i_star])


def propagate_one_orbit(method="rk4", use_j2=False, dt=60.0):
    orbit = make_reference_orbital_state()
    mu = EarthConstants.mu_e
    r_mag = np.linalg.norm(orbit.R)
    t_orbit = 2.0 * np.pi * np.sqrt(r_mag**3 / mu)
    steps = int(t_orbit / dt)
    dt = t_orbit / steps

    positions = np.zeros((steps + 1, 3))
    positions[0] = orbit.R

    for i in range(steps):
        if method == "rk4":
            orbit = orbit.propagate_orbit_rk4(dt, use_j2)
        elif method == "euler":
            orbit = orbit.propagate_orbit(dt, use_j2)
        else:
            raise ValueError(f"Unknown method: {method}")
        positions[i + 1] = orbit.R

    return dt, positions


def test_orbit_dynamics_matches_two_body_without_j2():
    state = make_reference_orbital_state()

    r_dot, v_dot = state.orbit_dynamics(J2_perturbation_on=False)

    expected_v_dot = -EarthConstants.mu_e * state.R / np.linalg.norm(state.R) ** 3
    assert np.allclose(r_dot, state.V)
    assert np.allclose(v_dot, expected_v_dot)


def test_propagate_orbit_advances_time_by_dt():
    state = make_reference_orbital_state()
    dt = 30.0

    propagated = state.propagate_orbit(dt=dt, J2_perturbation_on=True, fast=True)

    assert np.isclose(propagated.J2000, state.J2000 + dt / TimeConstants.cent2sec)


def test_propagate_orbit_rk4_advances_time_by_dt():
    state = make_reference_orbital_state()
    dt = 30.0

    propagated = state.propagate_orbit_rk4(dt=dt, J2_perturbation_on=True, fast=True)

    assert np.isclose(propagated.J2000, state.J2000 + dt / TimeConstants.cent2sec)


def test_rk4_orbit_closes_without_j2():
    _, positions = propagate_one_orbit(method="rk4", use_j2=False, dt=60.0)

    _, d_min = closest_approach(positions[0], positions)
    assert d_min < 1.0


def test_rk4_is_more_accurate_than_euler_for_one_orbit():
    _, rk4_positions = propagate_one_orbit(method="rk4", use_j2=False, dt=120.0)
    _, euler_positions = propagate_one_orbit(method="euler", use_j2=False, dt=120.0)

    _, rk4_err = closest_approach(rk4_positions[0], rk4_positions)
    _, euler_err = closest_approach(euler_positions[0], euler_positions)

    assert rk4_err < euler_err


def test_orbit_dynamics_jacobians_match_finite_difference():
    state = make_random_orbital_state(seed=11)

    def rfun(c):
        probe = make_reference_orbital_state()
        probe.R = np.array(c[:3], dtype=float)
        probe.V = np.array(c[3:], dtype=float)
        return probe.orbit_dynamics(J2_perturbation_on=True)[0]

    def vfun(c):
        probe = make_reference_orbital_state()
        probe.R = np.array(c[:3], dtype=float)
        probe.V = np.array(c[3:], dtype=float)
        return probe.orbit_dynamics(J2_perturbation_on=True)[1]

    x = state.R.tolist() + state.V.tolist()
    jr_num = np.array(nd.Jacobian(rfun)(x))
    jv_num = np.array(nd.Jacobian(vfun)(x))

    drd_dr, drd_dv, dvd_dr, dvd_dv = state.orbit_dynamics_jacobians(J2_perturbation_on=True)
    assert np.allclose(jr_num, np.hstack([drd_dr, drd_dv]))
    assert np.allclose(jv_num, np.hstack([dvd_dr, dvd_dv]))


def test_propagate_orbit_rk4_jacobians_match_finite_difference():
    state = make_random_orbital_state(seed=22)
    dt = 1.0

    def rfun(c):
        probe = make_reference_orbital_state()
        probe.R = np.array(c[:3], dtype=float)
        probe.V = np.array(c[3:], dtype=float)
        return probe.propagate_orbit_rk4(dt=dt, J2_perturbation_on=True).R

    def vfun(c):
        probe = make_reference_orbital_state()
        probe.R = np.array(c[:3], dtype=float)
        probe.V = np.array(c[3:], dtype=float)
        return probe.propagate_orbit_rk4(dt=dt, J2_perturbation_on=True).V

    x = state.R.tolist() + state.V.tolist()
    jr_num = np.array(nd.Jacobian(rfun)(x))
    jv_num = np.array(nd.Jacobian(vfun)(x))

    drd_dr, drd_dv, dvd_dr, dvd_dv = state.propagate_jacobians_rk4(dt=dt, J2_perturbation_on=True)
    analytic_r = np.hstack([drd_dr, drd_dv])
    analytic_v = np.hstack([dvd_dr, dvd_dv])

    assert np.allclose(jr_num, analytic_r)
    assert np.allclose(jv_num, analytic_v)


def test_copy_returns_independent_state():
    state = make_reference_orbital_state()

    copied = state.copy()
    copied.R[0] += 1.0
    copied.V[1] += 1.0

    assert not np.allclose(copied.R, state.R)
    assert not np.allclose(copied.V, state.V)
    assert np.isclose(copied.J2000, state.J2000)


def test_average_interpolates_core_fields_linearly():
    state0 = make_reference_orbital_state()
    state1 = state0.propagate_orbit_rk4(dt=60.0, J2_perturbation_on=True, fast=True)

    averaged = state0.average(state1, ratio=0.25)

    assert np.isclose(averaged.J2000, 0.75 * state0.J2000 + 0.25 * state1.J2000)
    assert np.allclose(averaged.R, 0.75 * state0.R + 0.25 * state1.R)
    assert np.allclose(averaged.V, 0.75 * state0.V + 0.25 * state1.V)
    assert np.allclose(averaged.S, 0.75 * state0.S + 0.25 * state1.S)
    assert np.allclose(averaged.B, 0.75 * state0.B + 0.25 * state1.B)
    assert np.allclose(averaged.rho, 0.75 * state0.rho + 0.25 * state1.rho)


def test_to_dict_and_from_dict_roundtrip_core_state():
    state = make_reference_orbital_state()

    restored = state.from_dict(state.to_dict(), ephem=state.ephem, density_model=state.density_model, fast=True)

    assert np.isclose(restored.J2000, state.J2000)
    assert np.allclose(restored.R, state.R)
    assert np.allclose(restored.V, state.V)
    assert np.allclose(restored.S, state.S)
    assert np.allclose(restored.B, state.B)
    assert np.allclose(restored.rho, state.rho)
