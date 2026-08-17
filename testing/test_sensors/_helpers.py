import numpy as np

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.state import State


def make_orbital_state(R=None, V=None, B=None, S=None):
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]) if R is None else np.asarray(R, dtype=float),
        V=np.array([0.0, 8.0, 0.0]) if V is None else np.asarray(V, dtype=float),
        B=np.array([1.0e-5, 2.0e-5, -1.5e-5]) if B is None else np.asarray(B, dtype=float),
        S=np.array([1.5e8, 1.0e7, -2.0e7]) if S is None else np.asarray(S, dtype=float),
        rho=0.0,
        fast=True,
    )


def make_state(q=None, w=None, h=None):
    q = np.array([1.0, 0.0, 0.0, 0.0]) if q is None else np.asarray(q, dtype=float)
    w = np.zeros(3) if w is None else np.asarray(w, dtype=float)
    return State(w=w, q=q, h=() if h is None else h)
