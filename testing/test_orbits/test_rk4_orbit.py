import numpy as np
import pytest

import ADCS
from ADCS.helpers.math_helpers import rot_mat
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.disturbances import Drag_Disturbance, GeometryConfig, GeometryFace, SRP_Disturbance


class ConstantAtmosphere(ADCS.Atmosphere):
    def __init__(self, rho):
        self.rho = rho

    def density(self, position_eci_km, datetime_utc=None, geodetic_lla=None):
        position = np.asarray(position_eci_km)
        return self.rho if position.ndim == 1 else np.full(position.shape[0], self.rho)

    def interpolate(self, altitude_km):
        return self.rho


def make_setup(*, rho=0.0, disturbances=()):
    environment = ADCS.Environment(atmosphere=ConstantAtmosphere(rho))
    os0 = ADCS.Orbital_State(
        J2000=0.22, R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]), environment=environment,
    )
    sat = ADCS.Satellite(mass=2.0, J_0=np.eye(3), disturbances=list(disturbances))
    x = ADCS.State.from_array([0, 0, 0, 1, 0, 0, 0])
    return os0, sat, x


def test_rk4_orbit_includes_earth_sun_and_moon_gravity():
    os0, sat, x = make_setup()
    orbit = ADCS.RK4Orbit(os0, zonal_J=0)
    derivative = orbit._derivative(0.0, np.r_[os0.R, os0.V], x, sat, os0)
    ephemeris = os0.environment.ephemeris
    expected = -EarthConstants.mu_e * os0.R / np.linalg.norm(os0.R) ** 3
    for body, mu in (("sun", EarthConstants.mu_sun), ("moon", EarthConstants.mu_moon)):
        body_r = ephemeris.position_eci(body, os0.J2000, apparent=False)
        expected += mu * (
            (body_r - os0.R) / np.linalg.norm(body_r - os0.R) ** 3
            - body_r / np.linalg.norm(body_r) ** 3
        )
    assert np.allclose(derivative[:3], os0.V)
    assert np.allclose(derivative[3:], expected, rtol=1e-12, atol=1e-15)


def test_drag_force_changes_acceleration_with_attitude():
    drag = Drag_Disturbance(GeometryConfig([
        GeometryFace(area=1.0, centroid=np.zeros(3), normal=np.array([0, 1, 0]), CD=2.0),
    ]))
    os0, sat, x_front = make_setup(rho=1e-7, disturbances=[drag])
    x_back = ADCS.State.from_array([0, 0, 0, 0, 0, 0, 1])
    orbit = ADCS.RK4Orbit(os0)
    y = np.r_[os0.R, os0.V]
    front = orbit._derivative(0.0, y, x_front, sat, os0)[3:]
    back = orbit._derivative(0.0, y, x_back, sat, os0)[3:]
    force_eci = rot_mat(x_front.q) @ sat.dist_forces(x_front, os0)
    assert np.allclose(front - back, force_eci / (sat.mass * 1000.0), atol=1e-15)
    assert front[1] < back[1]


def test_srp_force_enters_orbit_acceleration():
    os0, _, x = make_setup()
    sun_direction = os0.S - os0.R
    sun_direction /= np.linalg.norm(sun_direction)
    srp = SRP_Disturbance(GeometryConfig([
        GeometryFace(area=100.0, centroid=np.zeros(3), normal=sun_direction, eta_a=1.0),
    ]))
    sat = ADCS.Satellite(mass=2.0, J_0=np.eye(3), disturbances=[srp])
    baseline = ADCS.Satellite(mass=2.0, J_0=np.eye(3))
    orbit = ADCS.RK4Orbit(os0)
    y = np.r_[os0.R, os0.V]
    difference = orbit._derivative(0.0, y, x, sat, os0)[3:] - orbit._derivative(0.0, y, x, baseline, os0)[3:]
    expected = rot_mat(x.q) @ sat.dist_forces(x, os0) / (sat.mass * 1000.0)
    assert np.linalg.norm(expected) > 0
    assert np.allclose(difference, expected, rtol=1e-8, atol=1e-15)


def test_simulate_advances_coupled_orbit_after_attitude_changes():
    drag = Drag_Disturbance(GeometryConfig([
        GeometryFace(area=1.0, centroid=np.zeros(3), normal=np.array([0, 1, 0]), CD=2.0),
    ]))
    os0, sat, _ = make_setup(rho=1e-7, disturbances=[drag])
    x0 = ADCS.State.from_array([0, 0, 1.0, 1, 0, 0, 0])
    orbit = ADCS.RK4Orbit(os0)
    held_attitude = orbit.predict(x0, x0, sat, 1.0)

    results = ADCS.simulate(x=x0, satellite=sat, orbit=orbit, dt=1.0, tf=1.0)

    assert len(results.runs) == 1
    assert len(orbit.times) == 2
    assert np.linalg.norm(orbit.get_os(orbit.max_time()).V - held_attitude.V) > 1e-5
    with pytest.raises(ValueError, match="advance it sequentially"):
        orbit.get_os(orbit.max_time() + 1.0 / 36525.0)


def test_advance_stores_one_state_and_predict_does_not():
    os0, sat, x = make_setup()
    orbit = ADCS.RK4Orbit(os0)
    predicted = orbit.predict(x, x, sat, 10.0)
    assert len(orbit.times) == 1

    advanced = orbit.advance(x, x, sat, 10.0)

    assert len(orbit.times) == 2
    assert np.allclose(advanced.R, predicted.R)
    assert np.allclose(advanced.V, predicted.V)
    assert advanced.J2000 > os0.J2000


def test_simulate_accepts_precomputed_orbit_option():
    os0, sat, x = make_setup()
    orbit = ADCS.Orbit(
        os0, end_time=os0.J2000 + 2.0 * TimeConstants.sec2cent,
        dt=1.0, verbose=False,
    )

    results = ADCS.simulate(x=x, satellite=sat, orbit=orbit, dt=1.0, tf=1.0)

    assert len(results.runs) == 1
