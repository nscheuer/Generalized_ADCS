"""
Smoke coverage for the Monte-Carlo pipeline simulate_mc()
(critique pass).

simulate_mc() generates num_runs configs and executes them through
MonteCarloRunner (a real ProcessPoolExecutor with pickling of the
satellite/os0/results). Nothing in the suite exercised it -- the whole MC
path (config sampling, multiprocess fan-out, pickle round-trip, results
aggregation) was uncovered. It also consumes EstimatedSatellite.from_
satellite for the auto-built estimated model, so this additionally guards
that the (now deep-copying) auto-builder survives the pickle/subprocess
boundary.

Smoke + structural consistency: deterministic config (no samplers), 2
runs, 1 worker, tiny horizon -> a SimulationResults with exactly 2 runs of
finite, unit-quaternion state of the expected shape. GREEN on origin/main
(PR #37 model, test-only).
"""

import numpy as np
import pytest

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.helpers.math_constants import MathConstants
from ADCS.mc.simulate_mc import simulate_mc

pytestmark = pytest.mark.slow
_UV = MathConstants.unitvecs


def test_simulate_mc_runs_and_aggregates():
    ephem = Ephemeris()
    sat = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                    actuators=[MTQ(axis=_UV[j], max_torque=0.1) for j in range(3)],
                    sensors=[MTM(axis=_UV[j]) for j in range(3)])
    os0 = Orbital_State(ephem=ephem, J2000=0.22,
                        R=-7000.0 * np.array([0, np.sqrt(.5), np.sqrt(.5)]),
                        V=np.array([8.0, 0.0, 0.0]),
                        B=np.array([0.0, 0.1, 0.0]),
                        S=np.array([1e5 + 1, 0.0, 0.0]), rho=5e-12)
    x = np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0]])

    res = simulate_mc(x=x, satellite=sat, os0=os0, dt=1.0, tf=8.0,
                      num_runs=2, max_workers=1, base_seed=0)

    assert len(res.runs) == 2
    for r in res.runs:
        sh = np.asarray(r.state_hist, float)
        assert sh.ndim == 2 and sh.shape[0] >= 5 and sh.shape[1] == sat.state_len
        assert np.all(np.isfinite(sh))
        qn = np.linalg.norm(sh[:, 3:7], axis=1)
        np.testing.assert_allclose(qn, 1.0, atol=1e-3)

    # Indexing/iteration over the aggregated results must work.
    assert res[0] is res.runs[0]
    assert len(list(iter(res))) == 2
