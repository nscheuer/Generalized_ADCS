r"""
Single-satellite aerodynamics through ``simulate()``.

When an AeroModel is supplied, simulate() co-integrates the orbit with attitude
in the loop (operator-split drag + lift) instead of using the precomputed
gravity orbit. This must (a) leave the orbit unchanged when aero is off, (b)
measurably perturb the orbit when on, and (c) agree with a one-agent
Constellation run (the formation path) on the same scenario.
"""

import numpy as np

from ADCS.CONOPS.goals import No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.formation import SatelliteAgent, Constellation
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.aero import AeroModel
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.simulate import simulate

EPHEM = Ephemeris()


def _sat():
    return Satellite(mass=4.0, J_0=np.diagflat([0.02, 0.03, 0.04]))


def _aero():
    faces = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    areas = np.array([0.06, 0.06, 0.12, 0.12, 0.04, 0.04])
    return AeroModel(faces, areas, Cn=2.6, Ct=0.3)


def _os0():
    # No explicit rho: both simulate() and the Constellation's batched
    # environment derive density from the shared DensityModel (~2.3e-11 here),
    # so the orbit comparison is apples-to-apples.
    return Orbital_State(ephem=EPHEM, J2000=0.22,
                         R=np.array([6678.0, 0.0, 0.0]),
                         V=np.array([0.0, 7.726, 0.0]))


def _x0():
    return np.concatenate([np.zeros(3), normalize(np.array([1.0, 0.2, -0.1, 0.05]))])


def test_simulate_aero_perturbs_orbit_vs_gravity_only():
    dt, tf = 10.0, 1500.0
    off = simulate(x=_x0(), satellite=_sat(), goal=No_Goal(), os0=_os0(), dt=dt, tf=tf)[0]
    on = simulate(x=_x0(), satellite=_sat(), goal=No_Goal(), os0=_os0(), dt=dt, tf=tf, aero_model=_aero())[0]
    R_off = off.os_hist[-1].R
    R_on = on.os_hist[-1].R
    assert np.linalg.norm(R_on - R_off) > 1e-3


def test_simulate_aero_matches_one_agent_constellation():
    dt, tf = 10.0, 1000.0
    sim = simulate(x=_x0(), satellite=_sat(), goal=No_Goal(), os0=_os0(), dt=dt, tf=tf, aero_model=_aero())[0]

    agent = SatelliteAgent(x=_x0(), satellite=_sat(), goal_list=GoalList({0.22: No_Goal()}),
                           sat_id=0, aero_model=_aero())
    con = Constellation([agent], [_os0()], dt=dt, tf=tf, aero=True, verbose=False).run()[0]

    sim_R = np.vstack([os.R for os in sim.os_hist])
    con_R = np.vstack([os.R for os in con.os_hist])
    assert sim_R.shape == con_R.shape
    # Same operator-split RK4 orbit + equivalent batched/per-sat environment.
    assert np.allclose(sim_R, con_R, rtol=1e-7, atol=1e-6)

    sim_state = np.asarray(sim.state_hist, dtype=float)
    con_state = np.asarray(con.state_hist, dtype=float)
    assert np.allclose(sim_state, con_state, rtol=1e-6, atol=1e-7)


def test_simulate_without_aero_is_gravity_precomputed_path():
    # Sanity: the default (no aero) still produces a finite, normalized history.
    dt, tf = 10.0, 200.0
    res = simulate(x=_x0(), satellite=_sat(), goal=No_Goal(), os0=_os0(), dt=dt, tf=tf)[0]
    states = np.asarray(res.state_hist, dtype=float)
    assert states.shape[0] == int(tf / dt)
    assert np.all(np.isfinite(states))
    assert np.allclose(np.linalg.norm(states[:, 3:7], axis=1), 1.0, atol=1e-3)
