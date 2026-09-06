"""
End-to-end bridge test for ``SALTRO`` (the SALTRO-backed planner
in Generalized_ADCS, at ``ADCS/controller/saltro/saltro_controller.py``).

Before this file, the SALTRO bridge layer had **zero test coverage** in
Generalized_ADCS. This test exercises the full bridge plumbing: build a
Generalized_ADCS-side ``Satellite`` (with ``RW`` / ``MTM`` / boresight),
wrap in a ``SALTRO`` controller, and verify ``calculate_trajectory()``
returns a sensible result.

Scope
-----

This file deliberately tests **bridge integration**, not SALTRO optimizer
properties. A failure here could be in:

* The Generalized_ADCS ``SALTRO_planner_settings`` -> ``saltro_py``
  C++ settings translation.
* The Python ``Satellite`` actuator / sensor -> SALTRO C++ Satellite
  build in ``saltro_controller.py``.
* The orbit / goal sampling per timestep.
* The control / state tensor layout returned by ``saltro_py.trajOpt``.
* The SALTRO C++ optimizer itself.

SALTRO optimizer *properties* (warmstart determinism, axial symmetry
preservation, at-goal-commands-zero, ...) are tested separately in the
**SALTRO repo** at ``SALTRO/tests/unit/optimizer/`` using ``saltro_py``
directly (no Generalized_ADCS dependency, see PR nscheuer/SALTRO#TBD).
That split keeps SALTRO testable as a standalone optimizer library
without backward-depending on Generalized_ADCS, and keeps this file
focused on the bridge layer.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# SALTRO availability check
try:
    import saltro_py  # noqa: F401
    SALTRO_AVAILABLE = True
except ImportError:
    SALTRO_AVAILABLE = False

if not SALTRO_AVAILABLE:
    pytest.skip("saltro_py not available (build SALTRO first)",
                allow_module_level=True)

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.saltro.SALTRO_pass_settings import PassConfig
from ADCS.controller.saltro.SALTRO_planner_settings import PlannerSettings
from ADCS.controller.saltro.saltro_controller import SALTRO
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.satellite.satellite import Satellite


@pytest.mark.slow
def test_saltro_bridge_smoke():
    """End-to-end bridge plumbing test.

    Builds a Generalized_ADCS-side ``Satellite`` with a single y-axis
    RW, wraps it in a ``SALTRO`` controller, runs a small slew, and
    verifies the returned trajectory satisfies basic sanity checks:

    * Plans without raising.
    * State and control tensors have matching time dimensions.
    * No NaN / Inf in the result.
    * Quaternion unit-norm preserved (the C++ integrator is supposed to
      renormalise).
    * Controls within the configured actuator limit (catches the bridge
      losing the ``u_max`` value somewhere in the translation).

    If any of these break, the bridge layer or the SALTRO C++ optimizer
    has a real regression. The specific OPTIMIZER properties
    (warmstart determinism, symmetry, at-goal behaviour) are tested in
    the SALTRO repo directly; this test is just the integration smoke.
    """
    ephem = Ephemeris()
    rw = RW(axis=np.array([0.0, 1.0, 0.0]), max_torque=1.0,
            J=1e-6, h=0.0, h_max=10.0)
    sat = Satellite(mass=4.0, J_0=np.diagflat([1.0, 1.0, 1.0]),
                    actuators=[rw],
                    sensors=[MTM(axis=np.array([0.0, 0.0, 1.0]))],
                    boresight=np.array([0.0, 0.0, 1.0]))
    ps = PlannerSettings(est_sat=sat, passes=[PassConfig(dt=1.0)])
    ctrl = SALTRO(est_sat=sat, planner_settings=ps)

    # Small y-axis tilt + zero rate. Slew back to identity.
    q0 = np.array([np.cos(0.05), 0.0, np.sin(0.05), 0.0])
    q0 = q0 / np.linalg.norm(q0)
    x0 = np.concatenate([[0.0, 0.0, 0.0], q0, [0.0]])
    os0 = Orbital_State(ephem=ephem, J2000=0.22,
                        R=7000.0 * np.array([1.0, 0.0, 0.0]),
                        V=np.array([0.0, 7.5, 0.0]),
                        B=np.array([0.1, 0.0, 0.0]),
                        S=np.array([1e5, 0.0, 0.0]),
                        rho=0.0)

    traj = ctrl.calculate_trajectory(
        t_start=0.22, duration=30.0, x_0=x0, os_0=os0,
        goals=GoalList({0.22: Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0]))}),
        verbose=False,
    )

    states = np.asarray(traj.states)
    controls = np.asarray(traj.controls)

    # Shape sanity
    assert states.shape[0] >= 7, (
        f"State should have at least (omega, q): {states.shape}"
    )
    assert controls.shape[0] >= 1, (
        f"Need at least one control row: {controls.shape}"
    )
    assert states.shape[1] == controls.shape[1], (
        f"States {states.shape} and controls {controls.shape} should "
        f"share a time dimension"
    )

    # No NaN / Inf
    assert np.all(np.isfinite(states)), "States contain NaN/Inf"
    assert np.all(np.isfinite(controls)), "Controls contain NaN/Inf"

    # Quaternion unit-norm preserved
    q_norms = np.linalg.norm(states[3:7, :], axis=0)
    assert np.allclose(q_norms, 1.0, atol=1e-6), (
        f"Quaternion norm drift: {np.max(np.abs(q_norms - 1.0)):.3e}"
    )

    # Controls within the configured u_max = 1.0
    assert np.max(np.abs(controls)) <= 1.0 + 1e-6, (
        f"Controls exceed u_max=1.0: peak {np.max(np.abs(controls)):.4f}"
    )
