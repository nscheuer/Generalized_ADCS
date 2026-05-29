import numpy as np
import pytest

from ADCS.estimators.orbit_estimators import Orbit_GPS
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import GPS


R0 = np.array([7000.0, 1200.0, -800.0])
V0 = np.array([1.1, 7.4, 2.0])


def make_setup(std: float = 0.0):
    ephem = Ephemeris()
    true_state = Orbital_State(ephem=ephem, J2000=0.22, R=R0.copy(), V=V0.copy())
    gps = GPS(noise=Noise(noise=np.zeros(6), std_noise=np.full(6, std)))
    real_satellite = Satellite(sensors=[gps])
    estimated_satellite = EstimatedSatellite.from_satellite(real_satellite)
    template = Orbital_State(ephem=ephem, J2000=0.22, R=R0.copy(), V=V0.copy())
    estimator = Orbit_GPS(est_sat=estimated_satellite, J2000=true_state.J2000, os_template=template)
    return true_state, gps, estimator


def test_orbit_gps_noiseless_position_roundtrips_to_truth():
    true_state, gps, estimator = make_setup(std=0.0)
    measurement = gps.clean_reading(x=None, os=true_state)
    estimate = estimator.update(GPS_measurements=[measurement], J2000=true_state.J2000)
    np.testing.assert_allclose(estimate.os.R, true_state.R, rtol=0, atol=1e-6)


def test_orbit_gps_noiseless_velocity_roundtrips_to_truth():
    true_state, gps, estimator = make_setup(std=0.0)
    measurement = gps.clean_reading(x=None, os=true_state)
    estimate = estimator.update(GPS_measurements=[measurement], J2000=true_state.J2000)
    np.testing.assert_allclose(estimate.os.V, true_state.V, rtol=0, atol=1e-6)


def test_orbit_gps_empty_measurement_list_returns_prior():
    _, _, estimator = make_setup()
    estimate = estimator.update(GPS_measurements=[], J2000=0.22)
    assert estimate is estimator.os_hat


def test_orbit_gps_requires_at_least_one_gps_sensor():
    ephem = Ephemeris()
    template = Orbital_State(ephem=ephem, J2000=0.22, R=R0.copy(), V=V0.copy())
    estimated_satellite = EstimatedSatellite.from_satellite(Satellite(sensors=[]))
    with pytest.raises(ValueError):
        Orbit_GPS(est_sat=estimated_satellite, J2000=0.22, os_template=template)


def test_orbit_gps_position_only_measurement_recovers_position():
    true_state, _, estimator = make_setup(std=0.0)
    estimate = estimator.update(GPS_measurements=[np.asarray(true_state.ECEF, dtype=float)], J2000=true_state.J2000)
    np.testing.assert_allclose(estimate.os.R, true_state.R, rtol=0, atol=1e-6)
