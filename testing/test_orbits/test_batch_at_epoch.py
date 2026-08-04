r"""
Phase 2: ``Orbital_State.batch_at_epoch`` builds many satellites' environment at
one shared epoch in a single batched Skyfield/ppigrf pass. It must be
field-for-field equivalent to constructing each ``Orbital_State`` individually
(the expensive per-satellite path), while computing the time-only environment
(ECI<->ECEF frame, Sun) once and sharing it.
"""

import numpy as np
import pytest

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State

EPHEM = Ephemeris()
J2000 = 0.22

FIELDS = ["R", "V", "ECEF", "geocentric", "S", "B", "LLA",
          "ECI2ENUmat", "_R_eci2ecef", "_R_ecef2eci", "_ecef_to_geo",
          "_n_ecef", "_svec", "_tvec"]


def _random_states(n, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(n, 3))
    R = R / np.linalg.norm(R, axis=1, keepdims=True) * rng.uniform(6800.0, 7600.0, (n, 1))
    V = rng.normal(size=(n, 3)) * 0.1 + np.array([0.0, 7.4, 0.0])
    return R, V


@pytest.mark.parametrize("n", [1, 5, 25])
def test_batch_matches_per_satellite_construction(n):
    R, V = _random_states(n, seed=n)
    batch = Orbital_State.batch_at_epoch(R, V, J2000, EPHEM)
    assert len(batch) == n
    for i in range(n):
        single = Orbital_State(ephem=EPHEM, J2000=J2000, R=R[i], V=V[i])
        b = batch[i]
        for attr in FIELDS:
            a1 = np.asarray(getattr(single, attr), dtype=float)
            a2 = np.asarray(getattr(b, attr), dtype=float)
            assert np.allclose(a1, a2, rtol=1e-9, atol=1e-12), f"{attr} mismatch (sat {i})"
        assert abs(single.rho - b.rho) < 1e-15
        assert bool(single.is_sunlit()) == bool(b._sunlit)


def test_batch_shares_one_sun_vector_and_frame():
    R, V = _random_states(6, seed=42)
    batch = Orbital_State.batch_at_epoch(R, V, J2000, EPHEM)
    S0 = batch[0].S
    F0 = batch[0]._R_eci2ecef
    for b in batch[1:]:
        assert np.array_equal(b.S, S0)              # Sun is time-only -> shared
        assert np.array_equal(b._R_eci2ecef, F0)    # ECI<->ECEF frame shared


def test_batch_states_are_independent_objects():
    R, V = _random_states(3, seed=1)
    batch = Orbital_State.batch_at_epoch(R, V, J2000, EPHEM)
    batch[0].R[0] += 100.0
    assert batch[1].R[0] != batch[0].R[0]
    # mutating one state's position must not alias another's arrays
    assert batch[0].R is not batch[1].R


def test_batch_states_are_propagatable():
    # The states must support the normal orbit propagation API.
    R, V = _random_states(2, seed=2)
    batch = Orbital_State.batch_at_epoch(R, V, J2000, EPHEM)
    nxt = batch[0].propagate_orbit_rk4(10.0)
    assert np.all(np.isfinite(nxt.R)) and np.all(np.isfinite(nxt.V))
