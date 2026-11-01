"""simulate_mc must propagate with the models carried by os0.environment."""

import numpy as np
import pytest

from ADCS.environment.atmosphere import SMADAtmosphere
from ADCS.environment.environment import Environment
from ADCS.mc.simulate_mc import simulate_mc
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.state import State

SENTINEL_RHO = 1.234e-9


class ConstantAtmosphere(SMADAtmosphere):
    """Density no built-in model produces, so any substitution is visible."""

    def density(self, position_eci_km, datetime_utc=None, geodetic_lla=None):
        position = np.asarray(position_eci_km, dtype=float)
        return np.full(position.shape[:-1], SENTINEL_RHO) if position.ndim > 1 else SENTINEL_RHO

    def interpolate(self, altitude_km):
        altitude = np.asarray(altitude_km, dtype=float)
        return np.full(altitude.shape, SENTINEL_RHO) if altitude.ndim else SENTINEL_RHO


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="workers rebuild os0 from a dict with DE421 and default models")
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
