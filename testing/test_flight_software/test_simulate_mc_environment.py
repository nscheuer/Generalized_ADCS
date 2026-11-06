"""simulate_mc must propagate with the models carried by os0.environment."""

import numpy as np
import pytest
import json

from ADCS.environment.atmosphere import SMADAtmosphere
from ADCS.environment.environment import Environment
from ADCS.mc.simulate_mc import simulate_mc
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.state import State

SENTINEL_RHO = 1.234e-9


class MarkerEphemeris:
    def __init__(self):
        self.marker = "custom ephemeris"

    def to_json_config(self):
        return {"marker": self.marker}

    @classmethod
    def from_json_config(cls, config):
        restored = cls()
        restored.marker = config["marker"]
        return restored


class ConstantAtmosphere(SMADAtmosphere):
    """Density no built-in model produces, so any substitution is visible."""

    def density(self, position_eci_km, datetime_utc=None, geodetic_lla=None):
        position = np.asarray(position_eci_km, dtype=float)
        return np.full(position.shape[:-1], SENTINEL_RHO) if position.ndim > 1 else SENTINEL_RHO

    def interpolate(self, altitude_km):
        altitude = np.asarray(altitude_km, dtype=float)
        return np.full(altitude.shape, SENTINEL_RHO) if altitude.ndim else SENTINEL_RHO

    def to_json_config(self):
        return {"sentinel_rho": SENTINEL_RHO}

    @classmethod
    def from_json_config(cls, config):
        return cls()


def test_simulate_mc_propagates_with_os0_environment():
    os0 = Orbital_State(
        J2000=0.26, R=np.array([6878.0, 0.0, 0.0]), V=np.array([0.0, 7.6, 0.0]),
        environment=Environment(atmosphere=ConstantAtmosphere()),
    )
    assert os0.rho == SENTINEL_RHO

    result = simulate_mc(
        x=State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0]),
        satellite=Satellite(J_0=np.eye(3)),
        os0=os0, dt=1.0, tf=4.0, num_runs=1, max_workers=1, base_seed=0,
    )

    rho = [state.rho for state in result.runs[0].os_hist]
    np.testing.assert_allclose(rho, SENTINEL_RHO)


def test_environment_json_round_trip_keeps_custom_atmosphere():
    environment = Environment(atmosphere=ConstantAtmosphere())

    restored = Environment.from_json(environment.to_json())

    assert isinstance(restored.atmosphere, ConstantAtmosphere)
    assert restored.atmosphere.interpolate(500.0) == SENTINEL_RHO


def test_environment_json_round_trip_keeps_custom_ephemeris():
    environment = Environment(ephemeris=MarkerEphemeris())

    restored = Environment.from_json(environment.to_json())

    assert isinstance(restored.ephemeris, MarkerEphemeris)
    assert restored.ephemeris.marker == "custom ephemeris"


def test_environment_json_contains_model_descriptions_not_pickled_objects():
    payload = json.loads(Environment().to_json())

    assert payload["ephemeris"]["kind"] == "builtin"
    assert payload["atmosphere"]["kind"] == "builtin"
    assert all("data" not in component for component in payload.values() if isinstance(component, dict))
