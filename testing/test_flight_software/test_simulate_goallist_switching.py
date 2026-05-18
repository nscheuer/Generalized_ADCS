"""
Time-varying GoalList switching inside simulate() (critique pass).

simulate() accepts a GoalList and, every step, selects the active goal via
goal_list.get_active_goal(J2000_k). No test exercised a GoalList with more
than one goal through simulate(), so the goal-switching machinery (the
piecewise-constant schedule + the per-step target logging) was uncovered.

Independent reference: two hand-specified ECI_Goal directions and the
GoalList's own switch time -- the logged target history must equal goal A's
fixed ECI vector before the switch and goal B's after, with exactly one
transition.
"""

import numpy as np
import pytest

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.simulate import simulate

_UV = MathConstants.unitvecs


def test_simulate_switches_active_goal_per_goallist_schedule():
    ephem = Ephemeris()
    t0 = 0.22
    dt, tf = 1.0, 60.0
    t_switch = t0 + 30.0 * TimeConstants.sec2cent      # halfway

    vec_a = normalize(np.array([1.0, 0.0, 0.0]))
    vec_b = normalize(np.array([0.0, 1.0, 0.0]))
    goals = GoalList({t0: ECI_Goal(vec_a.copy()),
                      t_switch: ECI_Goal(vec_b.copy())},
                     time_units="centuries")

    sat = Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                    actuators=[MTQ(axis=_UV[j], max_torque=0.1) for j in range(3)],
                    sensors=[MTM(axis=_UV[j]) for j in range(3)])
    os0 = Orbital_State(ephem=ephem, J2000=t0,
                        R=-7000.0 * np.array([0, np.sqrt(.5), np.sqrt(.5)]),
                        V=np.array([8.0, 0.0, 0.0]),
                        B=np.array([0.0, 0.1, 0.0]),
                        S=np.array([1e5 + 1, 0.0, 0.0]), rho=5e-12)
    x = np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0]])

    res = simulate(x=x, satellite=sat, goal=goals, os0=os0, dt=dt, tf=tf)
    # target_hist rows are the to_ref vector-mode format [nan, vx, vy, vz];
    # the inertial direction is columns 1:4 (column 0 is the NaN sentinel).
    target = np.asarray(res[0].target_hist, float)
    assert target.ndim == 2 and target.shape[1] == 4 and target.shape[0] > 10
    assert np.all(np.isnan(target[:, 0])), "expected the NaN vector-mode sentinel"
    dirs = target[:, 1:4]

    # ECI_Goal.to_ref is a constant inertial vector -> the target history is
    # exactly vec_a then vec_b with a single switch.
    u = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    matches_a = np.all(np.isclose(u, vec_a, atol=1e-9), axis=1)
    matches_b = np.all(np.isclose(u, vec_b, atol=1e-9), axis=1)
    assert np.all(matches_a | matches_b), "target is neither goal A nor B"
    assert matches_a[0] and matches_b[-1], "goal did not switch A -> B"
    # exactly one A->B transition (piecewise-constant schedule)
    transitions = int(np.sum(np.diff(matches_a.astype(int)) != 0))
    assert transitions == 1, f"expected one goal switch, saw {transitions}"
    # the switch happens at the scheduled time (within one step)
    sw = int(np.argmax(~matches_a))
    t_sw_actual = t0 + sw * dt * TimeConstants.sec2cent
    assert abs(t_sw_actual - t_switch) <= 1.5 * dt * TimeConstants.sec2cent
