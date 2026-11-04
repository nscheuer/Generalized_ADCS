import numpy as np
import pytest

from ADCS.orbits.keplerian_orbit import KeplerianOrbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants


def test_keplerian_orbit_returns_initial_state_and_advances_two_body_motion():
    initial = Orbital_State.from_keplerian(
        J2000=0.22, a=7000.0, e=0.01, i=45.0, raan=20.0,
        arg_periapsis=30.0, true_anomaly=10.0,
    )
    dt = 120.0
    orbit = KeplerianOrbit(initial, end_time=initial.J2000 + 600.0 * TimeConstants.sec2cent, dt=dt)
    at_start = orbit.get_os(initial.J2000)
    at_dt = orbit.get_os(initial.J2000 + dt * TimeConstants.sec2cent)
    expected = initial.propagate_orbit_rk4(dt, zonal_J=0)

    np.testing.assert_allclose(at_start.R, initial.R, rtol=0.0, atol=1e-10)
    # The reference is finite-step RK4, so allow its truncation error.
    np.testing.assert_allclose(at_dt.R, expected.R, rtol=0.0, atol=2e-3)
    np.testing.assert_allclose(at_dt.V, expected.V, rtol=0.0, atol=2e-6)


def test_keplerian_orbit_rejects_unbound_initial_state():
    initial = Orbital_State.from_position(J2000=0.22, R=[7000, 0, 0], V=[0, 12, 0])
    try:
        KeplerianOrbit(initial)
    except ValueError:
        pass
    else:
        raise AssertionError("unbound orbit should be rejected")


def test_keplerian_orbit_accepts_states_built_by_orbit():
    # States from Orbit.get_os, copy and average skip __init__; the elements
    # are computed lazily, so these must still work.
    from ADCS.orbits.orbit import Orbit
    from ADCS.orbits.keplerian_orbit import KeplerianOrbit
    initial = Orbital_State.from_keplerian(J2000=0.22, a=7000., e=0.01, i=45., raan=20.,
                                           arg_periapsis=30., true_anomaly=10.)
    base = Orbit(initial, end_time=initial.J2000 + 120 * TimeConstants.sec2cent, dt=60., verbose=False)
    mid = base.get_os(initial.J2000 + 60 * TimeConstants.sec2cent)
    assert "eccentricity" not in mid.__dict__
    orbit = KeplerianOrbit(mid, end_time=mid.J2000 + 60 * TimeConstants.sec2cent, dt=30., verbose=False)
    assert mid.eccentricity == pytest.approx(0.01, abs=2e-3)
    assert "eccentricity" in mid.__dict__
    for derived in (initial.copy(), initial.average(mid, 0.5)):
        assert np.isfinite(derived.semi_major_axis)
        assert set(derived.keplerian_elements) == set(initial.keplerian_elements)


@pytest.mark.parametrize("e", [0.9, 0.97, 0.99])
def test_high_eccentricity_states_follow_keplers_equation(e):
    # Just after periapsis, where a fixed few Newton steps from E = M had not converged.
    a = 1122.0 / (1.0 - e) + 6378.137
    initial = Orbital_State.from_keplerian(J2000=0.22, a=a, e=e, i=63.4, raan=40.0,
                                           arg_periapsis=270.0, true_anomaly=0.0)
    n = np.sqrt(initial.mu_e / a**3)
    times = np.linspace(0.0, 0.7 / n, 15)
    orbit = KeplerianOrbit(initial, end_time=initial.J2000 + times[-1] * TimeConstants.sec2cent,
                           dt=times[1] - times[0])
    for t in times[1:]:
        _, _, _, _, _, nu = orbit.get_os(initial.J2000 + t * TimeConstants.sec2cent).to_keplerian(degrees=False)
        E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu / 2.0), np.sqrt(1.0 + e) * np.cos(nu / 2.0))
        M = np.mod(E - e * np.sin(E), 2.0 * np.pi)
        assert M == pytest.approx(n * t, abs=1e-8)
