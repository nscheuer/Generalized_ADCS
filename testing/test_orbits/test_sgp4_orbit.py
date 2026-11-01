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
    expected = 0.5 * 2.461e-5 * (2 * 2.0 * 2.0 / 4) / 4.0
    np.testing.assert_allclose(orbit.bstar, expected, rtol=1e-12)


def test_manual_bstar_overrides_drag_geometry():
    orbit = SGP4Orbit(make_state(), satellite=make_satellite(), bstar=0.0)
    assert orbit.bstar == 0.0
