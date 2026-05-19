"""
Test coverage for Orbit_GPS, the GPS pass-through orbit estimator.

Orbit_GPS converts an ECEF GPS measurement straight to an ECI EstimatedOrbital_State
(no dynamics; Q=0). The test verifies that a noiseless GPS measurement round-trips
through the sensor/estimator frame conversion and recovers the true orbital state.
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import GPS
from ADCS.satellite_hardware.errors import Noise
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.estimators.orbit_estimators import Orbit_GPS

_R = np.array([7000.0, 1200.0, -800.0])
_V = np.array([1.1, 7.4, 2.0])


def _setup(std=0.0):
    ephem = Ephemeris()
    os_true = Orbital_State(ephem=ephem, J2000=0.22, R=_R.copy(), V=_V.copy())
    gps = GPS(noise=Noise(noise=np.zeros(6), std_noise=np.full(6, std)))
    real_sat = Satellite(sensors=[gps])
    est_sat = EstimatedSatellite.from_satellite(real_sat)
    template = Orbital_State(ephem=ephem, J2000=0.22, R=_R.copy(), V=_V.copy())
    est = Orbit_GPS(est_sat=est_sat, J2000=os_true.J2000, os_template=template)
    return os_true, gps, est


def test_orbit_gps_recovers_known_truth_from_noiseless_measurement():
    os_true, gps, est = _setup(std=0.0)
    meas = gps.clean_reading(x=None, os=os_true)          # [r_ecef, v_ecef]
    out = est.update(GPS_measurements=[meas], J2000=os_true.J2000)
    # Position round-trips exactly through ECEF<->ECI.
    np.testing.assert_allclose(out.os.R, os_true.R, rtol=0, atol=1e-6)
    # Velocity uses the codebase's rotation-only ECEF convention
    # (eci_to_ecef on the sensor side, ecef_to_eci here) -- self-consistent,
    # so it also round-trips to the true ECI velocity.
    np.testing.assert_allclose(out.os.V, os_true.V, rtol=0, atol=1e-6)


def test_orbit_gps_no_measurement_returns_prior_without_crashing():
    _, _, est = _setup()
    # No prior estimate yet + empty measurement list must be handled.
    out = est.update(GPS_measurements=[], J2000=0.22)
    assert out is est.os_hat


def test_orbit_gps_requires_a_gps_sensor():
    ephem = Ephemeris()
    template = Orbital_State(ephem=ephem, J2000=0.22, R=_R.copy(), V=_V.copy())
    est_sat = EstimatedSatellite.from_satellite(Satellite(sensors=[]))
    with pytest.raises(ValueError):
        Orbit_GPS(est_sat=est_sat, J2000=0.22, os_template=template)


def test_orbit_gps_position_only_measurement():
    os_true, _, est = _setup(std=0.0)
    # 3-element (position-only) ECEF measurement.
    r_ecef = os_true.ECEF
    out = est.update(GPS_measurements=[np.asarray(r_ecef, float)],
                     J2000=os_true.J2000)
    np.testing.assert_allclose(out.os.R, os_true.R, rtol=0, atol=1e-6)
