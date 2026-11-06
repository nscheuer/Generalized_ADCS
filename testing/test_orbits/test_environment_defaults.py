"""Environment defaults and ephemeris integration."""

import numpy as np
from skyfield.api import load

from ADCS.environment import Environment
from ADCS.environment.ephemeris import DE421
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants


def test_default_environment_uses_de421():
    environment = Environment()
    assert isinstance(environment.ephemeris, DE421)
    assert environment.ephemeris._packaged_ephemeris_path().name == "de421.bsp"
    assert environment.magnetic_field.name == "igrf_numba"
    assert len(environment.star_catalogue.stars) == 30


def test_orbit_uses_ephemeris_without_skyfield_planets():
    class SunOnlyEphemeris:
        ts = load.timescale()

        def sun_position_eci(self, j2000):
            return np.broadcast_to([1.5e8, 0.0, 0.0], np.shape(j2000) + (3,))

    environment = Environment(ephemeris=SunOnlyEphemeris())
    state = Orbital_State(
        J2000=0.2, R=np.array([6800.0, 0.0, 0.0]),
        V=np.array([0.0, 7.6, 0.0]), environment=environment,
    )
    orbit = Orbit(state, dt=10, end_time=state.J2000 + 20 * TimeConstants.sec2cent, verbose=False)
    assert all(item.is_sunlit() for item in orbit.states.values())
    assert all(item.environment is environment for item in orbit.states.values())
