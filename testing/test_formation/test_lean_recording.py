r"""
Phase 5: lean / strided recording for large constellations.

- record_stride subsamples the recorded history (linear memory reduction).
- lean mode stores float32 numeric histories and a compact orbit position/
  velocity (os_pos_hist/os_vel_hist) instead of full Orbital_State objects and
  covariances, while still matching full-fidelity values at the recorded steps.
"""

import math
import numpy as np

from ADCS.CONOPS.goals import No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.formation import SatelliteAgent, Constellation
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.disturbances import GG_Disturbance
from ADCS.satellite_hardware.satellite import Satellite

EPHEM = Ephemeris()


def _sat():
    return Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]), disturbances=[GG_Disturbance()])


def _os0():
    return Orbital_State(ephem=EPHEM, J2000=0.22, R=np.array([7000.0, 0.0, 1200.0]), V=np.array([0.0, 7.4, 0.6]))


def _x0():
    return np.concatenate([np.array([0.02, -0.015, 0.03]), normalize(np.array([1.0, 0.2, -0.1, 0.05]))])


def _agent():
    return SatelliteAgent(x=_x0(), satellite=_sat(), goal_list=GoalList({0.22: No_Goal()}), sat_id=0)


def test_record_stride_subsamples_history():
    dt, tf, stride = 5.0, 200.0, 4
    n_steps = int(tf / dt)  # 40
    out = Constellation([_agent()], [_os0()], dt=dt, tf=tf, record_stride=stride, verbose=False).run()
    states = np.asarray(out[0].state_hist, dtype=float)
    assert states.shape[0] == math.ceil(n_steps / stride)


def test_lean_mode_uses_float32_and_compact_orbit():
    dt, tf = 5.0, 100.0
    out = Constellation([_agent()], [_os0()], dt=dt, tf=tf, lean=True, verbose=False).run()
    run = out[0]
    # float32 numeric histories
    assert np.asarray(run.state_hist).dtype == np.float32
    assert np.asarray(run.sensor_hist).dtype == np.float32 if run.sensor_hist else True
    # compact orbit history instead of full Orbital_State objects + covariances
    assert run.os_hist is None
    assert run.state_cov_hist is None
    assert run.os_pos_hist is not None and run.os_vel_hist is not None
    assert np.asarray(run.os_pos_hist).shape[1] == 3


def test_lean_matches_full_fidelity_at_recorded_steps():
    dt, tf = 5.0, 150.0
    full = Constellation([_agent()], [_os0()], dt=dt, tf=tf, verbose=False).run()[0]
    lean = Constellation([_agent()], [_os0()], dt=dt, tf=tf, lean=True, verbose=False).run()[0]

    full_state = np.asarray(full.state_hist, dtype=float)
    lean_state = np.asarray(lean.state_hist, dtype=float)
    assert full_state.shape == lean_state.shape
    # float32 round-off only
    assert np.allclose(full_state, lean_state, rtol=1e-5, atol=1e-5)

    full_R = np.vstack([os.R for os in full.os_hist])
    lean_R = np.asarray(lean.os_pos_hist, dtype=float)
    assert np.allclose(full_R, lean_R, rtol=1e-5, atol=1e-3)


def test_lean_and_stride_compose():
    dt, tf, stride = 5.0, 200.0, 5
    n_steps = int(tf / dt)
    out = Constellation([_agent()], [_os0()], dt=dt, tf=tf, lean=True, record_stride=stride, verbose=False).run()
    run = out[0]
    assert np.asarray(run.state_hist).dtype == np.float32
    assert np.asarray(run.os_pos_hist).shape[0] == math.ceil(n_steps / stride)
