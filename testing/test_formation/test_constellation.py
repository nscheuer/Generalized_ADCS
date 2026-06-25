r"""
Phase 3: Constellation orchestrator + FormationWorld + formation-aware goals.

Key cross-validation: a single-satellite Constellation (incremental orbit RK4 +
batched per-epoch environment) reproduces the historical single-satellite
``simulate()`` (precomputed Orbit) to high precision, using a gravity-gradient
disturbance so attitude is coupled to the orbital position.
"""

import numpy as np
import pytest

from ADCS.CONOPS.goals import No_Goal, Relative_Pointing_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.formation import SatelliteAgent, Constellation, FormationWorld
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.disturbances import GG_Disturbance
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.simulate import simulate

EPHEM = Ephemeris()


def _os0(R=None, V=None):
    return Orbital_State(
        ephem=EPHEM, J2000=0.22,
        R=np.array([7000.0, 0.0, 1200.0]) if R is None else np.asarray(R, dtype=float),
        V=np.array([0.0, 7.4, 0.6]) if V is None else np.asarray(V, dtype=float),
    )


def _gg_sat():
    return Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), disturbances=[GG_Disturbance()])


def _x0():
    w = np.array([0.02, -0.015, 0.03])
    q = normalize(np.array([1.0, 0.2, -0.1, 0.05]))
    return np.concatenate([w, q])


# --------------------------------------------------------------------------- #
# Formation-aware goal geometry
# --------------------------------------------------------------------------- #
def test_relative_pointing_goal_points_at_neighbour():
    world = FormationWorld()
    world.update("self", R=[7000.0, 0.0, 0.0], V=[0.0, 7.5, 0.0])
    world.update("tgt", R=[7000.0, 50.0, 30.0], V=[0.1, 7.5, 0.0])

    goal = Relative_Pointing_Goal(world, target_id="tgt")
    os_self = _os0(R=[7000.0, 0.0, 0.0], V=[0.0, 7.5, 0.0])
    r_ref, w_ref = goal.to_ref(os_self)

    los = np.array([7000.0, 50.0, 30.0]) - np.array([7000.0, 0.0, 0.0])
    rel_v = np.array([0.1, 7.5, 0.0]) - np.array([0.0, 7.5, 0.0])
    assert np.allclose(r_ref[1:4], normalize(los))
    assert np.allclose(w_ref, np.cross(los, rel_v) / np.dot(los, los))


# --------------------------------------------------------------------------- #
# Single-satellite Constellation == simulate()
# --------------------------------------------------------------------------- #
def test_single_sat_constellation_matches_simulate():
    dt, tf = 5.0, 200.0
    x0 = _x0()
    os0 = _os0()

    # Reference: historical single-satellite path.
    ref = simulate(x=x0, satellite=_gg_sat(), goal=No_Goal(), os0=os0, dt=dt, tf=tf)[0]
    ref_state = np.asarray(ref.state_hist, dtype=float)

    # Constellation with one agent.
    agent = SatelliteAgent(
        x=x0, satellite=_gg_sat(),
        goal_list=GoalList({os0.J2000: No_Goal()}), sat_id=0,
    )
    con = Constellation([agent], [os0], dt=dt, tf=tf, verbose=False)
    out = con.run()
    con_state = np.asarray(out[0].state_hist, dtype=float)

    assert con_state.shape == ref_state.shape
    # Same RK4 gravity orbit + same batched environment + same attitude
    # integrator => trajectories agree to integrator/round-off precision.
    assert np.allclose(con_state, ref_state, rtol=1e-7, atol=1e-9)

    # Orbit positions match too.
    ref_R = np.vstack([os.R for os in ref.os_hist])
    con_R = np.vstack([os.R for os in out[0].os_hist])
    assert np.allclose(con_R, ref_R, rtol=1e-9, atol=1e-6)


# --------------------------------------------------------------------------- #
# End-to-end formation run
# --------------------------------------------------------------------------- #
def test_constellation_formation_pointing_runs_and_records_each_sat():
    dt, tf = 5.0, 100.0
    world = FormationWorld()

    # Three satellites in a slightly offset ring, each pointing at the next.
    offsets = [np.zeros(3), np.array([0.0, 40.0, 0.0]), np.array([0.0, 0.0, 40.0])]
    os0_list = [_os0(R=np.array([7000.0, 0.0, 1200.0]) + o) for o in offsets]
    n = len(os0_list)

    agents = []
    for i in range(n):
        goal = Relative_Pointing_Goal(world, target_id=(i + 1) % n)
        agents.append(SatelliteAgent(
            x=_x0(), satellite=_gg_sat(),
            goal_list=GoalList({os0_list[i].J2000: goal}), sat_id=i,
        ))

    con = Constellation(agents, os0_list, dt=dt, tf=tf, world=world, verbose=False)
    out = con.run()

    assert len(out.runs) == n
    assert out.run_ids == [0, 1, 2]
    for run in out.runs:
        states = np.asarray(run.state_hist, dtype=float)
        assert states.shape[0] == int(tf / dt)
        assert np.all(np.isfinite(states))
        # quaternions stay normalized
        assert np.allclose(np.linalg.norm(states[:, 3:7], axis=1), 1.0, atol=1e-3)
        # the recorded target direction is finite (relative pointing produced a ref)
        targets = np.asarray(run.target_hist, dtype=float)
        assert np.all(np.isfinite(targets[:, 1:4]))


def test_constellation_satellites_drift_apart_under_different_initial_velocity():
    # Two satellites with slightly different along-track speed separate over time.
    dt, tf = 10.0, 600.0
    os_a = _os0(R=[7000.0, 0.0, 0.0], V=[0.0, 7.546, 0.0])
    os_b = _os0(R=[7000.0, 0.0, 0.0], V=[0.0, 7.540, 0.0])
    agents = [
        SatelliteAgent(x=_x0(), satellite=_gg_sat(), goal_list=GoalList({os_a.J2000: No_Goal()}), sat_id="a"),
        SatelliteAgent(x=_x0(), satellite=_gg_sat(), goal_list=GoalList({os_b.J2000: No_Goal()}), sat_id="b"),
    ]
    out = Constellation(agents, [os_a, os_b], dt=dt, tf=tf, verbose=False).run()
    Ra = np.vstack([os.R for os in out[0].os_hist])
    Rb = np.vstack([os.R for os in out[1].os_hist])
    sep = np.linalg.norm(Ra - Rb, axis=1)
    assert sep[0] < 1e-6          # start co-located
    assert sep[-1] > sep[0] + 1.0  # measurably drift apart
