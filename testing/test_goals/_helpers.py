import numpy as np

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite import Satellite


def make_test_orbital_state(
    R=None,
    V=None,
    B=None,
    S=None,
    J2000=0.22,
):
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=J2000,
        R=np.array([7000.0, 0.0, 0.0]) if R is None else np.asarray(R, dtype=float),
        V=np.array([0.0, 7.5, 0.0]) if V is None else np.asarray(V, dtype=float),
        B=np.array([0.0, 0.0, 2.0e-5]) if B is None else np.asarray(B, dtype=float),
        S=np.array([1.5e8, 0.0, 0.0]) if S is None else np.asarray(S, dtype=float),
        rho=0.0,
    )


def make_multi_boresight_satellite():
    return Satellite(
        mass=4.0,
        J_0=np.diagflat([0.1, 0.1, 0.1]),
        boresight={
            "camera": np.array([0.0, 0.0, 1.0]),
            "solar_panel": np.array([1.0, 0.0, 0.0]),
            "antenna": np.array([0.0, 1.0, 0.0]),
        },
    )
