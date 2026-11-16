import numpy as np

from ADCS.orbits.sgp4_orbit import SGP4Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.disturbances import Drag_Disturbance
from ADCS.satellite_hardware.disturbances.helpers.geometry_config import GeometryConfig, GeometryFace
from ADCS.satellite_hardware.satellite.satellite import Satellite


def make_satellite():
    config = GeometryConfig([
        GeometryFace(area=2.0, centroid=np.zeros(3), normal=np.array([1., 0., 0.]), CD=2.0),
        GeometryFace(area=2.0, centroid=np.zeros(3), normal=np.array([-1., 0., 0.]), CD=2.0),
    ])
    return Satellite(mass=4.0, disturbances=[Drag_Disturbance(config)])


def make_state():
    return Orbital_State.from_keplerian(
        J2000=0.22, a=7000., e=0.01, i=45., raan=20.,
        arg_periapsis=30., true_anomaly=10.,
    )


def test_sgp4_uses_manual_bstar_and_propagates():
    initial = make_state()
    orbit = SGP4Orbit(initial, end_time=initial.J2000 + 600 * TimeConstants.sec2cent,
                      dt=60., bstar=0.0002)
    assert orbit.bstar == 0.0002
    propagated = orbit.get_os(initial.J2000 + 60 * TimeConstants.sec2cent)
    assert np.all(np.isfinite(propagated.R))
    assert np.all(np.isfinite(propagated.V))
    assert not np.allclose(propagated.R, initial.R)


def test_sgp4_derives_attitude_averaged_bstar_from_drag_geometry():
    sat = make_satellite()
    orbit = SGP4Orbit(make_state(), satellite=sat)
    expected = 0.5 * 0.15696615 * (2 * 2.0 * 2.0 / 4) / 4.0
    np.testing.assert_allclose(orbit.bstar, expected, rtol=1e-12)


def test_manual_bstar_overrides_drag_geometry():
    orbit = SGP4Orbit(make_state(), satellite=make_satellite(), bstar=0.0)
    assert orbit.bstar == 0.0


def test_sgp4_matches_initial_state_at_epoch():
    # The mean elements are fitted, so SGP4 at the epoch reproduces os0 to the
    # precision of the TLE format (tens of metres), and the first step is continuous.
    initial = make_state()
    orbit = SGP4Orbit(initial, end_time=initial.J2000 + 60 * TimeConstants.sec2cent, dt=10., bstar=0.0)
    assert orbit.fit_residual_km < 0.05
    after = orbit.get_os(initial.J2000 + 10 * TimeConstants.sec2cent)
    expected_step = np.linalg.norm(initial.V) * 10.0
    assert abs(np.linalg.norm(after.R - initial.R) - expected_step) < 0.1
    # The fitted TLE differs from the naive osculating one.
    assert orbit.tle != initial.to_TLE()


def test_sgp4_fit_handles_circular_equatorial_orbit():
    initial = Orbital_State.from_keplerian(J2000=0.22, a=7000., e=0.0, i=0., raan=0.,
                                           arg_periapsis=0., true_anomaly=50.)
    orbit = SGP4Orbit(initial, bstar=0.0)
    assert orbit.fit_residual_km < 0.05


def test_sgp4_without_drag_disturbance_warns_and_uses_zero_bstar():
    import pytest
    with pytest.warns(UserWarning, match="no Drag_Disturbance"):
        orbit = SGP4Orbit(make_state(), satellite=Satellite(mass=4.0))
    assert orbit.bstar == 0.0


def test_sgp4_without_satellite_uses_zero_bstar():
    orbit = SGP4Orbit(make_state())
    assert orbit.bstar == 0.0


def test_sgp4_accepts_states_built_by_orbit():
    from ADCS.orbits.orbit import Orbit
    initial = make_state()
    base = Orbit(initial, end_time=initial.J2000 + 120 * TimeConstants.sec2cent, dt=60., verbose=False)
    mid = base.get_os(initial.J2000 + 60 * TimeConstants.sec2cent)
    orbit = SGP4Orbit(mid, bstar=0.0)
    assert orbit.fit_residual_km < 0.05


def _sentman_satellite(**sentman_kwargs):
    from ADCS.satellite_hardware.disturbances import Sentman_Drag_Disturbance
    config = GeometryConfig([
        GeometryFace(area=2.0, centroid=np.zeros(3), normal=np.array([1., 0., 0.]), CD=2.0),
        GeometryFace(area=2.0, centroid=np.zeros(3), normal=np.array([-1., 0., 0.]), CD=2.0),
    ])
    return Satellite(mass=4.0, disturbances=[Sentman_Drag_Disturbance(config, **sentman_kwargs)])


def test_sgp4_derives_bstar_from_sentman_drag():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "no drag disturbance" warning
        cold = SGP4Orbit(make_state(), satellite=_sentman_satellite(T_inf=1.0e-6, T_wall=0.0, alpha=1.0))
        hot = SGP4Orbit(make_state(), satellite=_sentman_satellite(T_inf=1500.0))
    # Cold-beam Sentman is the CD=2 flat plate, so B* matches Drag_Disturbance with CD=2.
    expected = 0.5 * 0.15696615 * (2 * 2.0 * 2.0 / 4) / 4.0
    np.testing.assert_allclose(cold.bstar, expected, rtol=1e-2)
    # Thermal motion and re-emission add drag over the cold beam.
    assert hot.bstar > cold.bstar


def test_sgp4_warns_when_the_tle_fit_fails():
    import pytest
    # One pass leaves the naive osculating TLE, kilometres from the initial state.
    with pytest.warns(UserWarning, match="TLE fit reproduces the initial position only"):
        orbit = SGP4Orbit(make_state(), bstar=0.0, fit_iterations=1)
    assert orbit.fit_residual_km > 0.1


def test_sgp4_does_not_warn_at_the_tle_precision_floor_in_geo():
    import warnings
    geo = Orbital_State.from_keplerian(J2000=0.22, a=42164., e=0.0002, i=0.1, raan=20.,
                                       arg_periapsis=30., true_anomaly=10.)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        orbit = SGP4Orbit(geo, bstar=0.0)
    # GEO cannot fit below the 1e-4 degree rounding of TLE angles (~74 m), so this is not a failure.
    assert orbit.fit_residual_km < 42164. * np.deg2rad(1.0e-4)


def test_sgp4_trajectory_is_continuous_from_its_first_node():
    import warnings
    initial = make_state()
    dt = 10.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a deliberately poor (one-pass) fit makes any jump obvious
        orbit = SGP4Orbit(initial, end_time=initial.J2000 + 3 * dt * TimeConstants.sec2cent, dt=dt,
                          bstar=0.0, fit_iterations=1)
    assert orbit.fit_residual_km > 0.1
    first = orbit.states[orbit.times[0]]
    # The first node is SGP4's own state at the epoch, not os0 ...
    t0 = orbit.environment.ephemeris.ts.tt_jd(orbit.times[0] * 36525.0 + 2451545.0)
    np.testing.assert_allclose(first.R, orbit._satellite.at(t0).position.km, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(first.R - initial.R), orbit.fit_residual_km, rtol=1e-9)
    # ... so the first step is as long as the others instead of carrying the fit residual.
    steps = [np.linalg.norm(orbit.states[b].R - orbit.states[a].R) for a, b in zip(orbit.times[:-1], orbit.times[1:])]
    assert max(steps) - min(steps) < 0.05 < orbit.fit_residual_km
