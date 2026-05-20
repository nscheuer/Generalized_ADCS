"""
Free-final-time approximation tests for the C++ trajectory planner.

The planner doesn't natively support free-final-time problems (the
optimizer takes a fixed horizon ``N`` and ``dt_tp``). The standard
workaround for time-optimal-style behaviour is to set the stage-cost
weights very high relative to terminal weights, so each step away from
the goal is heavily penalised and the planner is incentivised to reach
the goal as quickly as possible within the given horizon.

This file verifies that approximation works as intended:

1. ``test_planner_reaches_goal_well_inside_horizon`` -- with a 5x
   over-provisioned horizon and dominantly-stage cost weights, the
   planner must reach the goal vicinity in roughly ``N/5`` steps, not
   strung-out over the full horizon.
2. ``test_extended_horizon_same_head_trajectory`` -- adding more
   timesteps to the END of a problem should not change the planner's
   solution at the START (modulo iLQR-convergence noise). If it does,
   the planner isn't treating "extra time" as cheaply ignored.
3. ``test_stage_dominant_cost_drives_faster_convergence`` -- compare
   two cost-weight regimes (terminal-dominated vs stage-dominated)
   and verify the stage-dominated regime converges to the goal in
   fewer steps.

All three tests use the deterministic fixture (3 MTQ + ``bdot_on=1``,
see PR #72) so we get bit-exact reproducible trajectories.
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
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.satellite.satellite import Satellite


@pytest.fixture(scope="module")
def ephem():
    return Ephemeris()


def _build_planner(*, N, dt, Q_e, Q_omega, R_u, Q_e_N, Q_omega_N,
                   ephem, J=1.0, J_rw=1e-6, u_max=1.0):
    """Deterministic-fixture y-axis RW planner with given cost weights."""
    mtqs = [MTQ(axis=np.eye(3)[i], max_torque=1e-3) for i in range(3)]
    rw = RW(axis=np.array([0.0, 1.0, 0.0]), max_torque=u_max,
            J=J_rw, h=0.0, h_max=10.0)
    sat = Satellite(mass=4.0, J_0=np.diagflat([J, J, J]),
                    actuators=mtqs + [rw],
                    sensors=[MTM(axis=np.array([0.0, 0.0, 1.0]))],
                    boresight=np.array([0.0, 0.0, 1.0]))
    cw = CostWeights(
        angle=8.0 * Q_e, angle_N=8.0 * Q_e_N,
        ang_vel=2.0 * Q_omega, ang_vel_N=2.0 * Q_omega_N,
        control_mult=1.0, ang_cost_func_type=0,
    )
    ps = PlannerSettings(est_sat=sat, dt_tp=dt, dt_tvlqr=dt, bdot_on=1,
                         cost_main=cw, cost_second=cw, cost_tvlqr=cw)
    ps.verbosity = False
    ps.wmax = 5.0
    ps.rw_control_weight = 2.0 * R_u
    ps.mtq_control_weight = 1e10
    ps.control_limit_scale = 1.0
    ctrl = Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)
    os0 = Orbital_State(ephem=ephem, J2000=0.22,
                        R=7000.0 * np.array([1.0, 0.0, 0.0]),
                        V=np.array([0.0, 7.5, 0.0]),
                        B=np.array([0.1, 0.0, 0.0]),
                        S=np.array([1e5, 0.0, 0.0]),
                        rho=0.0)
    return ctrl, sat, os0


def _run(ctrl, os0, N, dt, w0_y):
    x0 = np.concatenate([[0.0, w0_y, 0.0],
                         [1.0, 0.0, 0.0, 0.0],
                         [0.0]])
    traj = ctrl.calculate_trajectory(
        t_start=0.22, duration=float(N) * dt, x_0=x0, os_0=os0,
        goals=GoalList({0.22: Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0]))}),
        verbose=False,
    )
    states = np.asarray(traj.states)
    return states, np.asarray(traj.controls)


def _y_axial_state_norm(states, k):
    """Distance from goal at step k: theta (rad) + |omega_y|."""
    qw, qy = states[3, k], states[5, k]
    theta = 2.0 * np.arctan2(abs(qy), abs(qw))
    omega = abs(states[1, k])
    return theta + omega


# ---------------------------------------------------------------------------
# T1: planner reaches goal well inside horizon under stage-dominant cost
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_planner_reaches_goal_well_inside_horizon(ephem):
    """With a 5x over-provisioned horizon and stage cost dominating
    terminal cost (the free-final-time approximation), the planner must
    drive the state into the goal neighbourhood within ~N/5 steps, not
    string the maneuver out over the full horizon.
    """
    dt = 1.0
    N = 150            # 5x what the maneuver would normally need
    w0_y = 0.01

    # Stage cost dominates terminal cost. With these weights every step
    # of "being away from goal" is expensive (Q_e * theta^2 + Q_omega *
    # omega^2 per step), and ``R_u`` is small so control is cheap.
    Q_e, Q_omega, R_u = 1e5, 1e5, 1.0
    Q_e_N, Q_omega_N = 1.0, 1.0     # ~no terminal incentive

    ctrl, _, os0 = _build_planner(
        N=N, dt=dt, Q_e=Q_e, Q_omega=Q_omega, R_u=R_u,
        Q_e_N=Q_e_N, Q_omega_N=Q_omega_N, ephem=ephem,
    )
    states, _ = _run(ctrl, os0, N, dt, w0_y)

    # Find first k where the state is within 5% of the initial distance.
    initial_dist = _y_axial_state_norm(states, 0)
    threshold = 0.05 * initial_dist
    first_within = None
    for k in range(N + 1):
        if _y_axial_state_norm(states, k) < threshold:
            first_within = k
            break

    assert first_within is not None, (
        f"Planner never got within 5% of initial distance over {N} steps; "
        f"initial = {initial_dist:.4e}, final = "
        f"{_y_axial_state_norm(states, N):.4e}"
    )
    # With 5x over-provisioned horizon + heavy stage cost, the planner
    # should reach this neighbourhood by N/5 = 30 steps.
    expected_max_steps = N // 3        # generous: N/3 instead of N/5
    assert first_within < expected_max_steps, (
        f"Planner took {first_within} steps to reach 5%-of-initial "
        f"neighbourhood; with stage-dominant cost (Q_e = {Q_e:.0e}, "
        f"Q_e_N = {Q_e_N:.0e}) we expected < {expected_max_steps}."
    )


# ---------------------------------------------------------------------------
# T2: extending the horizon doesn't change the head of the trajectory
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_extended_horizon_same_head_trajectory(ephem):
    """When the horizon is over-provisioned, doubling N must not change
    the planner's solution at the START of the trajectory: the planner
    should treat the extra time as free coasting time once the goal is
    reached, not as a reason to slow down the maneuver.
    """
    dt = 1.0
    w0_y = 0.01
    Q_e, Q_omega, R_u = 1e5, 1e5, 1.0
    Q_e_N, Q_omega_N = 1.0, 1.0

    ctrl1, _, os1 = _build_planner(
        N=80, dt=dt, Q_e=Q_e, Q_omega=Q_omega, R_u=R_u,
        Q_e_N=Q_e_N, Q_omega_N=Q_omega_N, ephem=ephem,
    )
    states_80, ctl_80 = _run(ctrl1, os1, 80, dt, w0_y)

    ctrl2, _, os2 = _build_planner(
        N=160, dt=dt, Q_e=Q_e, Q_omega=Q_omega, R_u=R_u,
        Q_e_N=Q_e_N, Q_omega_N=Q_omega_N, ephem=ephem,
    )
    states_160, ctl_160 = _run(ctrl2, os2, 160, dt, w0_y)

    # First 20 steps should match closely: the maneuver is essentially
    # complete by step ~30, so the head dynamics are identical.
    n_compare = 20
    head_state_diff = float(np.max(np.abs(
        states_80[:, :n_compare] - states_160[:, :n_compare]
    )))
    state_scale = max(np.max(np.abs(states_80[:, :n_compare])), 1e-9)

    # Allow 5% of state magnitude as tolerance -- catches the planner
    # changing its early-trajectory strategy depending on horizon, while
    # absorbing iLQR-convergence noise.
    assert head_state_diff < 0.05 * state_scale, (
        f"Head trajectory changes when horizon extended: "
        f"max|state diff| over first {n_compare} steps = "
        f"{head_state_diff:.3e}, peak state = {state_scale:.3e}, "
        f"ratio {head_state_diff / state_scale * 100:.1f}%."
    )


# ---------------------------------------------------------------------------
# T3: stage-dominant vs terminal-dominant -> different time-to-goal
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_stage_dominant_cost_drives_faster_convergence(ephem):
    """Two cost-weight regimes on the same problem:

    * stage-dominant: high Q_e, Q_omega; low Q_e_N, Q_omega_N
    * terminal-dominant: low Q_e, Q_omega; high Q_e_N, Q_omega_N

    The stage-dominant regime should reach the goal vicinity in FEWER
    steps than the terminal-dominant regime. Confirms that stage-cost
    weights actually drive faster convergence (the free-final-time
    approximation premise).
    """
    dt = 1.0
    N = 100
    w0_y = 0.02

    # Two regimes with the same TOTAL Q.
    ctrl_stage, _, os_stage = _build_planner(
        N=N, dt=dt,
        Q_e=1e5, Q_omega=1e5, R_u=1.0,
        Q_e_N=1.0, Q_omega_N=1.0,
        ephem=ephem,
    )
    states_stage, _ = _run(ctrl_stage, os_stage, N, dt, w0_y)

    ctrl_term, _, os_term = _build_planner(
        N=N, dt=dt,
        Q_e=1.0, Q_omega=1.0, R_u=1.0,
        Q_e_N=1e5, Q_omega_N=1e5,
        ephem=ephem,
    )
    states_term, _ = _run(ctrl_term, os_term, N, dt, w0_y)

    # Time to reach 5% of initial distance:
    def time_to_threshold(states, threshold):
        d0 = _y_axial_state_norm(states, 0)
        for k in range(N + 1):
            if _y_axial_state_norm(states, k) < threshold * d0:
                return k
        return N + 1

    t_stage = time_to_threshold(states_stage, 0.05)
    t_term = time_to_threshold(states_term, 0.05)
    assert t_stage < t_term, (
        f"Stage-dominant did not converge faster than terminal-dominant:\n"
        f"  stage  reaches 5% in {t_stage} steps\n"
        f"  term   reaches 5% in {t_term} steps"
    )
