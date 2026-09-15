"""The CG5 commutator-free Lie-group integrator's tableau and accuracy.

Row 3 of the ``a`` matrix shipped summing to 1.324 against ``c[3] = 0.324``,
so that stage was evaluated a full unit interval late and the method lost its
order: a 5 s attitude step under a 0.01 N m wheel torque was 7e-3 rad off a
fine reference where RK4 is 1e-6 off. Every legacy UKF sigma point went
through that branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.process_model import propagate_state
from ADCS.orbits import universal_constants as uc
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.state import State


def test_cg5_tableau_is_consistent():
    a, b, c = np.asarray(uc.CG5_a), np.asarray(uc.CG5_b), np.asarray(uc.CG5_c)
    np.testing.assert_allclose(a.sum(axis=1), c, rtol=0.0, atol=1.0e-12)
    assert b.sum() == pytest.approx(1.0, abs=1.0e-12)
    assert np.allclose(a, np.tril(a, -1)), "explicit method: strictly lower-triangular a"


def _attitude_error(p: State, q: State) -> float:
    return float(2.0 * np.arccos(min(1.0, abs(float(np.dot(p.q, q.q))))))


def test_cg5_converges_at_second_order():
    """Refining the step 10x must cut the error ~100x. With the inconsistent
    row the method was first order (a 10x cut) and 25-40x less accurate."""
    satellite = Satellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        actuators=[RW(axis=np.array([0.0, 0.0, 1.0]), max_torque=0.02, J=1.0e-3, h=0.0, h_max=1.0)],
    )
    orbital_state = Orbital_State(
        ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]), fast=True,
    )
    state = State(w=np.array([0.05, -0.02, 0.03]), q=np.array([0.9, 0.2, -0.3, 0.1]), h=np.zeros(1)).normalized()
    control = np.array([0.01])

    def run(steps: int, integrator: str) -> State:
        x = state
        for _ in range(steps):
            x = propagate_state(x, satellite, control, 5.0 / steps, orbital_state, orbital_state,
                                quaternion_integrator=integrator)
        return x

    reference = run(1000, "rk4")
    errors = [_attitude_error(run(steps, "cg5"), reference) for steps in (1, 10, 100)]
    assert errors[0] / errors[1] > 50.0, errors
    assert errors[1] / errors[2] > 50.0, errors
    assert errors[0] < 1.0e-2, errors
