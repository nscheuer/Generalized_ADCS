r"""
Phase 5: scale smoke test -- a many-satellite constellation with the full
feature stack (formation-pointing goals + attitude-coupled aero + higher-order
gravity + lean recording) runs end-to-end and returns one result per satellite.

Kept intentionally short (few steps) so it stays CI-friendly; the per-step cost
is dominated by the attitude integrator and scales ~linearly in N.
"""

import numpy as np

from ADCS.CONOPS.goals import Relative_Pointing_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.formation import SatelliteAgent, Constellation, FormationWorld
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.aero import AeroModel
from ADCS.satellite_hardware.disturbances import GG_Disturbance
from ADCS.satellite_hardware.satellite import Satellite

EPHEM = Ephemeris()


def _box_aero():
    faces = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    areas = np.array([0.06, 0.06, 0.12, 0.12, 0.04, 0.04])
    return AeroModel(faces, areas, Cn=2.4, Ct=0.4)


def test_many_satellite_constellation_runs_full_stack():
    n = 120
    dt, tf = 5.0, 20.0  # 4 steps
    rng = np.random.default_rng(0)
    world = FormationWorld()

    base = np.array([6778.0, 0.0, 0.0])
    agents, os0s = [], []
    for i in range(n):
        off = rng.normal(scale=2.0, size=3)
        off[0] = 0.0
        os0s.append(Orbital_State(
            ephem=EPHEM, J2000=0.22,
            R=base + off,
            V=np.array([0.0, 7.726, 0.0]) + rng.normal(scale=1e-3, size=3),
            rho=5e-11,
        ))
        sat = Satellite(mass=4.0, J_0=np.diagflat([0.02, 0.03, 0.04]),
                        disturbances=[GG_Disturbance()]).seed(i)
        x0 = np.concatenate([rng.normal(scale=0.01, size=3), normalize(rng.normal(size=4))])
        goal = Relative_Pointing_Goal(world, target_id=(i + 1) % n)
        agents.append(SatelliteAgent(
            x=x0, satellite=sat, goal_list=GoalList({0.22: goal}),
            sat_id=i, aero_model=_box_aero(),
        ))

    con = Constellation(agents, os0s, dt=dt, tf=tf, world=world,
                        aero=True, zonal_order=4, lean=True, verbose=False)
    out = con.run()

    assert len(out.runs) == n
    assert out.run_ids == list(range(n))
    for run in out.runs:
        states = np.asarray(run.state_hist, dtype=float)
        assert states.shape == (int(tf / dt), 7)
        assert np.all(np.isfinite(states))
        assert np.allclose(np.linalg.norm(states[:, 3:7], axis=1), 1.0, atol=1e-3)
