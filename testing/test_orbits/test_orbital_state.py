import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.orbits.orbit import Orbit

from testing.test_orbits._helpers import make_random_orbital_state, make_reference_orbital_state


def closest_approach(reference, trajectory, min_skip=1):
    d2 = np.sum((trajectory - reference) ** 2, axis=1)
    if min_skip > 0:
        d2[:min_skip] = np.inf
    i_star = np.argmin(d2)
    return i_star, np.sqrt(d2[i_star])


def central_jacobian(function, x, steps):
    """Independent central-difference Jacobian without state construction."""
    x = np.asarray(x, dtype=float)
    jacobian = np.empty((len(function(x)), len(x)))
    for column, step in enumerate(steps):
        offset = np.zeros_like(x)
        offset[column] = step
        jacobian[:, column] = (function(x + offset) - function(x - offset)) / (2.0 * step)
    return jacobian


def propagate_one_orbit(method="rk4", use_j2=False, dt=60.0):
    orbit = make_reference_orbital_state()
    mu = EarthConstants.mu_e
    r_mag = np.linalg.norm(orbit.R)
    t_orbit = 2.0 * np.pi * np.sqrt(r_mag**3 / mu)
    steps = int(t_orbit / dt)
    dt = t_orbit / steps

    if method == "rk4":
        end_time = orbit.J2000 + t_orbit * TimeConstants.sec2cent
        batch = Orbit(
            os0=orbit,
            end_time=end_time,
            dt=dt,
            zonal_J=2 if use_j2 else 0,
            verbose=False,
        )
        return dt, np.vstack([batch.states[t].R for t in batch.times])

    positions = np.zeros((steps + 1, 3))
    positions[0] = orbit.R

    for i in range(steps):
        if method == "euler":
            orbit = orbit.propagate_orbit(dt, zonal_J=2 if use_j2 else 0)
        else:
            raise ValueError(f"Unknown method: {method}")
        positions[i + 1] = orbit.R

    return dt, positions


def test_orbit_dynamics_matches_two_body_without_j2():
    state = make_reference_orbital_state()

    r_dot, v_dot = state.orbit_dynamics(zonal_J=0)

    expected_v_dot = -EarthConstants.mu_e * state.R / np.linalg.norm(state.R) ** 3
    assert np.allclose(r_dot, state.V)
    assert np.allclose(v_dot, expected_v_dot)


def test_propagate_orbit_advances_time_by_dt():
    state = make_reference_orbital_state()
    dt = 30.0

    propagated = state.propagate_orbit(dt=dt, zonal_J=2, fast=True)

    assert np.isclose(propagated.J2000, state.J2000 + dt / TimeConstants.cent2sec)


def test_propagate_orbit_rk4_advances_time_by_dt():
    state = make_reference_orbital_state()
    dt = 30.0

    propagated = state.propagate_orbit_rk4(dt=dt, zonal_J=2, fast=True)

    assert np.isclose(propagated.J2000, state.J2000 + dt / TimeConstants.cent2sec)


def test_rk4_orbit_closes_without_j2():
    _, positions = propagate_one_orbit(method="rk4", use_j2=False, dt=120.0)

    _, d_min = closest_approach(positions[0], positions)
    assert d_min < 1.0


def test_rk4_is_more_accurate_than_euler_for_one_orbit():
    _, rk4_positions = propagate_one_orbit(method="rk4", use_j2=False, dt=180.0)
    _, euler_positions = propagate_one_orbit(method="euler", use_j2=False, dt=180.0)

    _, rk4_err = closest_approach(rk4_positions[0], rk4_positions)
    _, euler_err = closest_approach(euler_positions[0], euler_positions)

    assert rk4_err < euler_err


def test_orbit_dynamics_jacobians_match_finite_difference():
    state = make_random_orbital_state(seed=11)

    def dynamics(x):
        r_dot, v_dot = state._orbit_dynamics_raw(
            x[:3], x[3:], state.mu_e, state.R_e, state.J2coeff, zonal_J=2, Jcoeffs=state.Jcoeffs
        )
        return np.concatenate((r_dot, v_dot))

    x = np.concatenate((state.R, state.V))
    numeric = central_jacobian(dynamics, x, steps=[1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6])
    drd_dr, drd_dv, dvd_dr, dvd_dv = state.orbit_dynamics_jacobians(zonal_J=2)
    analytic = np.block([[drd_dr, drd_dv], [dvd_dr, dvd_dv]])
    assert np.allclose(numeric, analytic, rtol=1e-6, atol=1e-10)


def test_propagate_orbit_rk4_jacobians_match_finite_difference():
    state = make_random_orbital_state(seed=22)
    dt = 1.0

    def rhs(_, x):
        r_dot, v_dot = state._orbit_dynamics_raw(
            x[:3], x[3:], state.mu_e, state.R_e, state.J2coeff, zonal_J=2, Jcoeffs=state.Jcoeffs
        )
        return np.concatenate((r_dot, v_dot))

    def high_accuracy_step(x):
        solution = solve_ivp(rhs, (0.0, dt), x, method="DOP853", rtol=1e-12, atol=1e-13)
        assert solution.success
        return solution.y[:, -1]

    x = np.concatenate((state.R, state.V))
    numeric = central_jacobian(high_accuracy_step, x, steps=[1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4])
    drd_dr, drd_dv, dvd_dr, dvd_dv = state.propagate_jacobians_rk4(dt=dt, zonal_J=2)
    analytic = np.block([[drd_dr, drd_dv], [dvd_dr, dvd_dv]])

    assert np.allclose(numeric, analytic, rtol=1e-6, atol=1e-8)


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
    state1 = state0.propagate_orbit_rk4(dt=60.0, zonal_J=2, fast=True)

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
