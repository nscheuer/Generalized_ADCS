import numpy as np

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
