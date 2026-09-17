from __future__ import annotations

import numpy as np
import pytest

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State


@pytest.fixture(scope="session")
def ephem() -> Ephemeris:
    """One real ephemeris shared by the complete orbit test suite."""
    return Ephemeris()


@pytest.fixture
def state(ephem: Ephemeris) -> Orbital_State:
    """A LEO state with supplied environment vectors to avoid unnecessary I/O."""
    return Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=np.array([7000.0, 1200.0, -800.0]),
        V=np.array([-1.1, 7.4, 2.0]),
        S=np.array([1.4e8, -2.0e7, 3.0e6]),
        B=np.array([2.0e-5, -1.0e-5, 3.0e-5]),
        rho=1.0e-12,
    )
