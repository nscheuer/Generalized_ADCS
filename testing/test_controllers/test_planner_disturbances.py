"""
Disturbance-aware planning tests for the C++ trajectory planner.

The planner exposes ``include_gg``, ``include_aero``, ``include_srp``,
``include_resdipole``, ``include_prop``, ``include_gendist`` flags on
``PlannerSettings`` that are *supposed* to make the planner pre-compensate
for the corresponding environmental or actuator-bias torques. Currently
**all of these flags are effectively dead** -- the planner produces nearly
identical trajectories regardless of whether they're set, even with large
disturbance magnitudes. Tracked in issue #76.

The tests in this file have two purposes:

1. **Document the current state** with strict-xfail assertions, so when
   #76 is fixed, the tests start XPASSing and we know to flip them green.
2. **Pin a regression tripwire**: if the dead-flag behaviour ever becomes
   active in a wrong way (e.g., disturbance accidentally injected
   unsigned), the test ``test_dead_flags_produce_no_planner_response``
   catches it.

The structure is symmetric: each test runs the planner twice with the
same problem, once with the flag off, once with it on. A working flag
must produce *meaningfully* different trajectories.
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
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.disturbances import (
    Prop_Disturbance,
    Dipole_Disturbance,
)
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.satellite.satellite import Satellite


@pytest.fixture(scope="module")
def ephem():
    return Ephemeris()


def _build_y_axis_sat(disturbances=None):
    """Y-axis RW + 3 dummy MTQs (deterministic-warmstart fixture, see PR #72)."""
    mtqs = [MTQ(axis=np.eye(3)[i], max_torque=1e-3) for i in range(3)]
    rw = RW(axis=np.array([0.0, 1.0, 0.0]), max_torque=1.0,
            J=1e-6, h=0.0, h_max=10.0)
    return Satellite(mass=4.0, J_0=np.diagflat([0.1, 0.1, 0.1]),
                     actuators=mtqs + [rw],
                     sensors=[MTM(axis=np.array([0.0, 0.0, 1.0]))],
                     boresight=np.array([0.0, 0.0, 1.0]),
                     disturbances=list(disturbances or []))


def _make_os(ephem):
    return Orbital_State(ephem=ephem, J2000=0.22,
                        R=7000.0 * np.array([1.0, 0.0, 0.0]),
                        V=np.array([0.0, 7.5, 0.0]),
                        B=np.array([0.1, 0.0, 0.0]),
                        S=np.array([1e5, 0.0, 0.0]),
                        rho=0.0)


def _run(sat, ephem, *, N=30, dt=1.0, **planner_settings_kwargs):
    cw = CostWeights(angle=8e3, angle_N=8e4, ang_vel=2e4, ang_vel_N=2e5,
                     control_mult=1.0, ang_cost_func_type=0)
    ps = PlannerSettings(est_sat=sat, dt_tp=dt, dt_tvlqr=dt, bdot_on=1,
                         cost_main=cw, cost_second=cw, cost_tvlqr=cw,
                         **planner_settings_kwargs)
    ps.verbosity = False
    ps.wmax = 1.0
    ps.rw_control_weight = 2.0
    ps.mtq_control_weight = 1e10
    ps.control_limit_scale = 1.0
    ctrl = Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)
    x0 = np.concatenate([[0.0, 0.01, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0]])
    traj = ctrl.calculate_trajectory(
        t_start=0.22, duration=float(N) * dt, x_0=x0, os_0=_make_os(ephem),
        goals=GoalList({0.22: Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0]))}),
        verbose=False,
    )
    controls = np.asarray(traj.controls)
    states = np.asarray(traj.states)
    # RW is the last actuator (after 3 dummy MTQs).
    return states, controls[3, :]


# ---------------------------------------------------------------------------
# Currently-dead flag detection (xfailed; flip to green when #76 lands)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.xfail(strict=False,
                   reason="tracking #76: include_prop=True is silently ignored "
                          "(prop_vals schedule never populated in "
                          "_propagate_environment)")
def test_include_prop_meaningfully_compensates_constant_torque(ephem):
    """A constant propulsion torque of 0.001 N*m on the y-axis over 30s
    integrates to 0.030 N*m*s of angular impulse. The planner with
    ``include_prop=True`` *should* pre-compensate by adding ~-0.030 N*m*s
    of RW impulse to cancel it. Currently observed: zero compensation.
    """
    disturbance = Prop_Disturbance(np.array([0.0, 0.001, 0.0]))
    sat_with = _build_y_axis_sat([disturbance])
    sat_without = _build_y_axis_sat([disturbance])

    _, u_off = _run(sat_without, ephem, include_prop=False)
    _, u_on = _run(sat_with, ephem, include_prop=True)

    delta_impulse = float(u_on.sum() - u_off.sum())
    # Expected: ~-0.030 N*m*s (a 30s constant 0.001 N*m torque)
    # Observed today: ~0 (the flag is dead)
    assert abs(delta_impulse) > 0.5 * 0.030, (
        f"include_prop=True did not produce meaningful compensation: "
        f"delta_impulse = {delta_impulse:.5e}, expected ~ -0.030 N*m*s. "
        f"See issue #76."
    )


@pytest.mark.slow
@pytest.mark.xfail(strict=False,
                   reason="tracking #76: include_resdipole=True is silently "
                          "ignored on planned trajectory (broader scope than "
                          "the original prop_vals bug)")
def test_include_resdipole_meaningfully_compensates_dipole_torque(ephem):
    """A residual dipole of ``[0, 0, 1] A*m^2`` in B = ``[0.1, 0, 0]`` T
    produces a body torque ``m x B = [0, 0.1, 0]`` N*m -- a sizable
    y-axis torque the planner should compensate via the RW. Currently
    observed: ~0 compensation despite the large torque.
    """
    sat_with = _build_y_axis_sat([Dipole_Disturbance(np.array([0.0, 0.0, 1.0]))])
    sat_without = _build_y_axis_sat([Dipole_Disturbance(np.array([0.0, 0.0, 1.0]))])

    _, u_off = _run(sat_without, ephem, include_resdipole=False)
    _, u_on = _run(sat_with, ephem, include_resdipole=True)

    delta_impulse = float(u_on.sum() - u_off.sum())
    # Expected: ~-3.0 N*m*s (a 30s constant 0.1 N*m torque)
    assert abs(delta_impulse) > 0.5 * 3.0, (
        f"include_resdipole=True did not produce meaningful compensation: "
        f"delta_impulse = {delta_impulse:.5e}, expected ~ -3.0 N*m*s. "
        f"See issue #76."
    )


@pytest.mark.slow
def test_include_gg_meaningfully_changes_trajectory(ephem):
    """Gravity-gradient torque ``3 mu/r^3 * (n x J*n)`` is normally small
    at LEO altitudes but should produce a measurable change in the
    planner's trajectory for non-isotropic inertia + off-nadir attitude.

    With ``J = diag(1, 5, 3)`` at 0.4 rad pitch the y-axis GG torque is
    of order ~3e-7 N*m, but its integrated effect over a long horizon
    (600s) gives a measurable trajectory diff. This is the one
    disturbance flag we've verified is actually live; ``include_prop``
    and ``include_resdipole`` appear dead (see issue #76 and the two
    xfailed tests above).
    """
    sat_with = _build_y_axis_sat()
    sat_with.J_0 = np.diagflat([1.0, 5.0, 3.0])
    sat_without = _build_y_axis_sat()
    sat_without.J_0 = np.diagflat([1.0, 5.0, 3.0])

    _, u_off = _run(sat_without, ephem, N=600, include_gg=False)
    _, u_on = _run(sat_with, ephem, N=600, include_gg=True)

    rel_diff = float(np.max(np.abs(u_on - u_off)) /
                     max(float(np.max(np.abs(u_off))), 1e-12))
    assert rel_diff > 0.01, (
        f"include_gg=True produced negligible trajectory change: "
        f"max|u_on - u_off| / max|u_off| = {rel_diff:.3e}. See issue #76."
    )


# ---------------------------------------------------------------------------
# Regression tripwire: dead flags should at least produce IDENTICAL output
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_dead_flags_produce_no_planner_response_today(ephem):
    """While the disturbance flags are dead (issue #76), the planner
    output with ``include_X=True`` should be **identical** (or very nearly
    so) to ``include_X=False``. If a future change accidentally injects
    a disturbance the wrong way (e.g., unsigned, or without proper
    cost-gradient information), this test catches it.

    When #76 is fixed and the flags become live, this test will start
    failing -- update or remove it as part of the fix.
    """
    disturbance = Prop_Disturbance(np.array([0.0, 0.001, 0.0]))
    sat_with = _build_y_axis_sat([disturbance])
    sat_without = _build_y_axis_sat([disturbance])

    _, u_off = _run(sat_without, ephem, include_prop=False)
    _, u_on = _run(sat_with, ephem, include_prop=True)
    diff = float(np.max(np.abs(u_on - u_off)))
    # The flag is dead, so the planner should produce numerically
    # identical (bit-exact) controls. A small drift (~1e-12) is acceptable;
    # anything large indicates the flag is partially live in a wrong way.
    assert diff < 1e-8, (
        f"include_prop=True changed trajectory by {diff:.3e} -- the flag "
        f"appears to be partially live (issue #76 is supposed to be dead). "
        f"Investigate before assuming this is a fix."
    )
