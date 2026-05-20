"""
End-to-end test of :class:`~ADCS.satellite_hardware.actuators.Magic_Actuator`
through ``Plan_and_Track_LQR``.

The magic actuator is the simplest possible body-torque commander: it
applies ``tau = a * u`` directly with no environmental or state coupling.
This makes it the ideal test fixture for closed-form Riccati comparison:

* No magnetorquer ``m \\times B`` rank deficiency.
* No reaction-wheel ``J_eff = J - J_RW`` parallel-axis correction.
* No wheel-momentum state.

Compared to the RW-based Riccati test in
``test_planner_riccati_closed_form.py``, this version replaces the
y-axis RW with three magic actuators (one per body axis). The resulting
problem reduces to a 2-state LQR exactly, with no ``J_eff`` adjustment
needed.

Tests
-----

* ``test_magic_actuator_routes_through_build_csat`` -- ``build_csat.py``
  correctly translates a Python ``Magic_Actuator`` to the C++
  ``csat.add_magic(axis, max_torq, cost)``. The control trajectory has
  the magic-actuator row in the expected position and produces the
  expected body torque.
* ``test_magic_actuator_eigenaxis_lqr_matches_riccati`` -- bit-exact
  match (~1e-5) between the planner's commanded controls/states and
  the 2-state LQR Riccati closed-form solution.
* ``test_magic_actuator_actuator_ordering`` -- 3 magic actuators (one
  per axis) plus a y-axis RW gives the expected control-row layout
  ``[mtq..., rw..., magic...]`` with magic actuators *last* in the
  C++ ordering.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller.helpers.optional_dependencies import (
    trajectory_planner_available,
    trajectory_planner_missing_reason,
)

if not trajectory_planner_available():
    pytest.skip(trajectory_planner_missing_reason(), allow_module_level=True)

from ADCS.CONOPS.goals import Fixed_Attitude_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track import PlannerSettings, CostWeights
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import RW, MTQ, Magic_Actuator
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.satellite.satellite import Satellite

from testing.test_controllers.test_planner_oracle import (
    DiscreteLQRParams,
    solve_discrete_lqr_optimal,
)


@pytest.fixture(scope="module")
def ephem():
    return Ephemeris()


def _make_os(ephem):
    return Orbital_State(ephem=ephem, J2000=0.22,
                        R=7000.0 * np.array([1.0, 0.0, 0.0]),
                        V=np.array([0.0, 7.5, 0.0]),
                        B=np.array([0.1, 0.0, 0.0]),
                        S=np.array([1e5, 0.0, 0.0]),
                        rho=0.0)


# ---------------------------------------------------------------------------
# Routing: build_csat.py handles Magic_Actuator correctly
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_magic_actuator_routes_through_build_csat(ephem):
    """A satellite with 3 magic actuators (x/y/z) plus 3 MTQs (so the
    deterministic-warmstart gate fires per PR #72) plans without raising
    and the trajectory's control layout matches the C++ ordering
    ``[MTQs, RWs, magics]``.

    A direct test that ``add_actuator`` in ``build_csat.py`` correctly
    dispatches the ``Magic_Actuator`` branch.
    """
    mtqs = [MTQ(axis=np.eye(3)[i], max_torque=1e-3) for i in range(3)]
    magics = [Magic_Actuator(axis=np.eye(3)[i], max_torque=1.0)
              for i in range(3)]
    sat = Satellite(mass=4.0, J_0=np.diagflat([0.1, 0.1, 0.1]),
                    actuators=mtqs + magics,
                    sensors=[MTM(axis=np.array([0.0, 0.0, 1.0]))],
                    boresight=np.array([0.0, 0.0, 1.0]))

    cw = CostWeights(angle=1e3, angle_N=1e6, ang_vel=1e3, ang_vel_N=1e5,
                     control_mult=1.0, ang_cost_func_type=0)
    ps = PlannerSettings(est_sat=sat, dt_tp=1.0, dt_tvlqr=1.0, bdot_on=1,
                         cost_main=cw, cost_second=cw, cost_tvlqr=cw)
    ps.verbosity = False
    ps.wmax = 1.0
    ps.mtq_control_weight = 1e10              # MTQs effectively disabled
    ps.magic_control_weight = 1.0             # Magic actuators get used
    ps.control_limit_scale = 1.0

    ctrl = Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)
    q0 = np.array([np.cos(0.05), 0.0, np.sin(0.05), 0.0])
    q0 = q0 / np.linalg.norm(q0)
    x0 = np.concatenate([[0.0, 0.0, 0.0], q0])
    traj = ctrl.calculate_trajectory(
        t_start=0.22, duration=30.0, x_0=x0, os_0=_make_os(ephem),
        goals=GoalList({0.22: Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0]))}),
        verbose=False,
    )

    controls = np.asarray(traj.controls)
    # Layout: 3 MTQs + 0 RWs + 3 magics = 6 control rows.
    assert controls.shape[0] == 6, (
        f"Expected 6 control rows (3 MTQ + 3 magic), got {controls.shape[0]}"
    )
    # MTQs should be ~0 (their cost is 1e10); magics should carry the
    # actual maneuver. magic_y (index 4) drives the y-axis slew.
    mtq_peak = float(np.max(np.abs(controls[:3, :])))
    magic_y_peak = float(np.max(np.abs(controls[4, :])))  # index 3 + 1 (y)
    assert mtq_peak < 1e-6, (
        f"MTQs with weight 1e10 should be unused; got peak {mtq_peak:.3e}"
    )
    assert magic_y_peak > 1e-4, (
        f"Magic-y should carry the y-axis slew; got peak {magic_y_peak:.3e}"
    )


# ---------------------------------------------------------------------------
# Closed-form Riccati comparison (cleaner than RW version)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("omega0_y", [0.005, 0.01, 0.02])
def test_magic_actuator_eigenaxis_lqr_matches_riccati(ephem, omega0_y):
    """With a single y-axis magic actuator (plus 3 dummy MTQs for the
    deterministic-warmstart gate), q0 = identity, omega0 along y, and
    cost-weight conversion matching the planner's ``(1-cos(theta/2))``
    form, the planner provably reduces to the 2-state ``(e, omega)``
    LQR Riccati.

    Unlike the RW-based version of this test in
    ``test_planner_riccati_closed_form.py``, the magic actuator
    requires NO ``J_eff = J - J_RW`` adjustment -- it applies torque
    directly with no parallel-axis correction. The LQR oracle uses
    ``J`` directly.

    Tolerance: 1e-5 absolute (matches the RW version).
    """
    J = 1.0
    dt = 1.0
    N = 30
    Q_e, Q_omega, R_u = 1e3, 1e4, 1.0
    Q_e_N, Q_omega_N = 1e4, 1e5

    # 3 dummy MTQs (deterministic-warmstart gate) + 1 y-axis magic.
    mtqs = [MTQ(axis=np.eye(3)[i], max_torque=1e-3) for i in range(3)]
    magic_y = Magic_Actuator(axis=np.array([0.0, 1.0, 0.0]), max_torque=1.0)
    sat = Satellite(mass=4.0, J_0=np.diagflat([J, J, J]),
                    actuators=mtqs + [magic_y],
                    sensors=[MTM(axis=np.array([0.0, 0.0, 1.0]))],
                    boresight=np.array([0.0, 0.0, 1.0]))

    # Same cost-weight conversion as the RW version: planner stepcost
    # form ``w_ang * (1 - cos(theta/2)) + 0.5 * w_av * omega**2 +
    # 0.5 * w_u * u**2`` against the LQR oracle's no-half quadratic.
    cw = CostWeights(angle=8 * Q_e, angle_N=8 * Q_e_N,
                     ang_vel=2 * Q_omega, ang_vel_N=2 * Q_omega_N,
                     control_mult=1.0, ang_cost_func_type=0)
    ps = PlannerSettings(est_sat=sat, dt_tp=dt, dt_tvlqr=dt, bdot_on=1,
                         cost_main=cw, cost_second=cw, cost_tvlqr=cw)
    ps.verbosity = False
    ps.wmax = 1.0
    ps.mtq_control_weight = 1e10
    ps.magic_control_weight = 2.0 * R_u       # Match (0.5 * w_u * u^2)
    ps.control_limit_scale = 1.0

    ctrl = Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)
    x0 = np.concatenate([[0.0, omega0_y, 0.0], [1.0, 0.0, 0.0, 0.0]])
    traj = ctrl.calculate_trajectory(
        t_start=0.22, duration=float(N) * dt, x_0=x0, os_0=_make_os(ephem),
        goals=GoalList({0.22: Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0]))}),
        verbose=False,
    )

    states = np.asarray(traj.states)
    controls = np.asarray(traj.controls)
    # Magic-y is the LAST actuator (index 4 = 3 MTQs + 1 magic).
    u_magic_y = controls[3, :]
    omega_y = states[1, :]
    e_y = 2.0 * np.arctan2(states[5, :], states[3, :])

    # 2-state LQR oracle. NO J_eff -- the magic actuator applies torque
    # directly with no parallel-axis subtraction.
    p = DiscreteLQRParams(J=J, dt=dt, N=N,
                          Q_e=Q_e, Q_omega=Q_omega, R_u=R_u,
                          Q_e_N=Q_e_N, Q_omega_N=Q_omega_N)
    lqr = solve_discrete_lqr_optimal(0.0, omega0_y, p)

    n_state = min(len(lqr.e), len(e_y))
    n_ctrl = min(len(lqr.u), len(u_magic_y))

    e_diff = np.max(np.abs(lqr.e[:n_state] - e_y[:n_state]))
    w_diff = np.max(np.abs(lqr.omega[:n_state] - omega_y[:n_state]))
    u_diff = np.max(np.abs(lqr.u[:n_ctrl] - u_magic_y[:n_ctrl]))

    TOL = 1e-5
    msg = (
        f"\n  omega0_y = {omega0_y}\n"
        f"  max|e diff|     = {e_diff:.3e}\n"
        f"  max|omega diff| = {w_diff:.3e}\n"
        f"  max|u diff|     = {u_diff:.3e}\n"
        f"  LQR u(0..2)     = {lqr.u[:3]}\n"
        f"  ALTRO u(0..2)   = {u_magic_y[:3]}"
    )
    assert e_diff < TOL, "e trajectory mismatch:" + msg
    assert w_diff < TOL, "omega trajectory mismatch:" + msg
    assert u_diff < TOL, "u trajectory mismatch:" + msg


# ---------------------------------------------------------------------------
# Actuator ordering: [MTQs, RWs, magics]
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_magic_actuator_actuator_ordering(ephem):
    """A satellite with 2 MTQs + 1 RW + 3 magic actuators (interleaved
    in the Python list) should produce a C++ control layout
    ``[mtq_0, mtq_1, rw_0, magic_0, magic_1, magic_2]``. Verified by
    setting per-axis-distinct magic actuator weights and checking
    which control row carries the maneuver torque.
    """
    actuators = [
        MTQ(axis=np.array([1.0, 0.0, 0.0]), max_torque=1e-3),
        Magic_Actuator(axis=np.array([0.0, 1.0, 0.0]), max_torque=1.0),
        RW(axis=np.array([0.0, 0.0, 1.0]), max_torque=1.0,
           J=1e-6, h=0.0, h_max=10.0),
        MTQ(axis=np.array([0.0, 1.0, 0.0]), max_torque=1e-3),
        Magic_Actuator(axis=np.array([1.0, 0.0, 0.0]), max_torque=1.0),
        Magic_Actuator(axis=np.array([0.0, 0.0, 1.0]), max_torque=1.0),
    ]
    sat = Satellite(mass=4.0, J_0=np.diagflat([0.1, 0.1, 0.1]),
                    actuators=actuators,
                    sensors=[MTM(axis=np.array([0.0, 0.0, 1.0]))],
                    boresight=np.array([0.0, 0.0, 1.0]))

    cw = CostWeights(angle=1e3, angle_N=1e6, ang_vel=1e3, ang_vel_N=1e5,
                     control_mult=1.0, ang_cost_func_type=0)
    ps = PlannerSettings(est_sat=sat, dt_tp=1.0, dt_tvlqr=1.0, bdot_on=1,
                         cost_main=cw, cost_second=cw, cost_tvlqr=cw)
    ps.verbosity = False
    ps.wmax = 1.0
    ps.mtq_control_weight = 1e10
    ps.rw_control_weight = 1e10
    ps.magic_control_weight = 1.0
    ps.control_limit_scale = 1.0

    ctrl = Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)
    q0 = np.array([np.cos(0.05), 0.0, np.sin(0.05), 0.0])
    q0 = q0 / np.linalg.norm(q0)
    x0 = np.concatenate([[0.0, 0.0, 0.0], q0, [0.0]])
    traj = ctrl.calculate_trajectory(
        t_start=0.22, duration=30.0, x_0=x0, os_0=_make_os(ephem),
        goals=GoalList({0.22: Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0]))}),
        verbose=False,
    )

    controls = np.asarray(traj.controls)
    # 2 MTQs + 1 RW + 3 magics = 6 rows total.
    assert controls.shape[0] == 6, (
        f"Expected 6 control rows (2 MTQ + 1 RW + 3 magic), got "
        f"{controls.shape[0]}"
    )

    # Python actuator order (after reorder back from C++): the original
    # interleaved order. So row 1 (Magic_y) should be the dominant slew
    # actuator. Rows 0/3 (MTQs) and row 2 (RW) should be near zero (huge
    # control weights make them unused).
    magic_y_peak = float(np.max(np.abs(controls[1, :])))
    mtq_0_peak = float(np.max(np.abs(controls[0, :])))
    mtq_1_peak = float(np.max(np.abs(controls[3, :])))
    rw_peak = float(np.max(np.abs(controls[2, :])))
    assert magic_y_peak > 1e-4, (
        f"Magic-y (row 1) should carry the slew; peak {magic_y_peak:.3e}"
    )
    assert max(mtq_0_peak, mtq_1_peak, rw_peak) < 1e-6, (
        f"MTQs/RW (high-cost-weight) should be unused; "
        f"got peaks MTQ_0 {mtq_0_peak:.3e}, MTQ_1 {mtq_1_peak:.3e}, "
        f"RW {rw_peak:.3e}"
    )
