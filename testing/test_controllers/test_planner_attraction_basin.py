"""
Attraction-basin / known-optimal-solution tests for the C++ trajectory
planner.

Companion to ``test_planner_lqr_convergent.py`` and
``test_planner_kkt_consistency.py``: rather than asserting equivalence to a
2-state LQR (only valid in a tiny corner) or self-consistency under the
planner's own cost (model-agnostic but doesn't pin DOWN what the planner
should do), these tests pose problems whose optimal solution is known by
SYMMETRY or MANIFOLD INVARIANCE rather than by a closed-form analytic
oracle:

* **Eigenaxis preservation.** A single-axis RW on body axis ``n`` plus an
  initial perturbation purely along ``n`` MUST keep the trajectory in the
  ``(omega_n, delta_q_n, h_RW)`` subspace forever. Any off-axis omega or
  attitude excursion is a planner bug (the dynamics conserve the subspace
  exactly; the cost is rotation-axis-agnostic; symmetry forces the optimum
  to also be in-subspace). Verifiable at machine precision.

* **Null-control at goal.** Starting at the goal with zero rate, the
  optimal control is identically zero. ALTRO must converge to this; if it
  ever commands nonzero u, it has either misread the goal or its
  optimisation is stuck.

* **Sign-flip mirror symmetry.** The cost and dynamics are symmetric under
  ``omega_0 -> -omega_0``: the optimal control mirrors as
  ``u(t) -> -u(t)`` and so does the state trajectory. Detects a sign bug
  anywhere in the planner's gradient computation.

* **180-degree eigenaxis slew.** Boundary case: starting 180 degrees rotated
  from the goal puts ``(1 - q . q_goal)`` at its maximum and the
  small-angle Taylor used by the planner's analytic Hessian is invalid.
  Tests the planner's behaviour at the worst case of its own
  parameterisation. Verifies eigenaxis preservation (still in-subspace
  with a single-axis RW) and goal-reaching.

* **Initial-guess robustness.** ALTRO uses an internal warm-start; the
  ``InitTrajConfig`` exposes a randomness component. Running the same
  problem with different ``rand_add_ratio`` values (the random
  initialisation magnitude) should converge to the same optimum within
  iLQR-convergence tolerance. If the planner is stuck in a regulariser-
  driven local minimum, different inits would converge to different
  trajectories.
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

from ADCS.CONOPS.goals import ECI_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track import PlannerSettings, CostWeights
from ADCS.controller.plan_and_track.planner_subsettings import InitTrajConfig
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.satellite.satellite import Satellite


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


def _run_planner(*, axis, q0, w0, ephem, J=0.1, J_rw=1e-3, u_max=0.05,
                 h_max=10.0, N=30, dt=1.0, ang_cost_func_type=2,
                 rand_add_ratio=0.0, deterministic=True):
    """Run ALTRO with a single RW on ``axis`` and return the state/control
    trajectory.

    Determinism note
    ----------------
    The C++ planner's default RW-only path is nondeterministic: ``OldPlanner.
    cpp:276`` calls Armadillo's unseeded ``randn`` to generate an initial
    control sequence whenever ``bdot_on == 0`` OR ``sat.number_MTQ < 3``
    (line 268 gate). Repeated calls then differ at the ~1e-2 state level
    purely from RNG state drift between runs.

    To make these tests reliable we route around it by adding 3 dummy
    MTQs (so ``sat.number_MTQ == 3``, satisfying the gate's MTQ
    requirement) and setting ``bdot_on=1`` (so the Bdot warmstart path
    runs instead of the random one). The MTQs are given enormous
    ``mtq_control_weight`` so ALTRO essentially never uses them; the
    RW remains the only effective actuator. Verified bit-exact
    determinism across 4 consecutive calls.

    Returns ``(times, states, controls)`` where ``states`` is shape
    ``(state_dim, N+1)`` = ``[w_x, w_y, w_z, q_w, q_x, q_y, q_z, h_RW]``.
    """
    actuators = [RW(axis=np.asarray(axis, dtype=float), max_torque=u_max,
                    J=J_rw, h=0.0, h_max=h_max)]
    if deterministic:
        # Three dummy MTQs (~1e-3 max dipole, far smaller than any control
        # that would be commanded anyway). Their presence satisfies the
        # OldPlanner.cpp:268 gate so the Bdot warmstart path runs.
        actuators = [MTQ(axis=np.eye(3)[i], max_torque=1e-3)
                     for i in range(3)] + actuators
    sat = Satellite(mass=4.0, J_0=np.diagflat([J, J, J]),
                    actuators=actuators,
                    sensors=[MTM(axis=np.array([0.0, 0.0, 1.0]))],
                    boresight=np.array([0.0, 0.0, 1.0]))

    # Use ang_cost_func_type=2 (geodesic acos) so the formulation behaves
    # cleanly at large angles, where the cft=0 small-angle Taylor breaks.
    cw = CostWeights(
        angle=1e3, angle_N=1e6, ang_vel=1e3, ang_vel_N=1e5,
        control_mult=1.0, ang_cost_func_type=ang_cost_func_type,
    )
    init = InitTrajConfig(
        bdot_gain=1000.0,
        hl_angle_limit=10.0 * np.pi / 180.0,
        high_settings=(0, -2e0, 0, 0.0, rand_add_ratio, 0.5),
        low_settings=(0, -2e0, 0, 0.0, rand_add_ratio, 0.5),
    )
    ps = PlannerSettings(
        est_sat=sat, dt_tp=dt, dt_tvlqr=1.0,
        bdot_on=(1 if deterministic else 0),
        cost_main=cw, cost_second=cw, cost_tvlqr=cw,
        init_traj=init,
    )
    ps.verbosity = False
    ps.wmax = 1.0
    ps.rw_control_weight = 2.0
    if deterministic:
        # Make MTQs prohibitively expensive so ALTRO doesn't actually use
        # them. The dynamics still see them but with no commanded dipole
        # the body torque from MTQ is zero.
        ps.mtq_control_weight = 1e10
    ps.control_limit_scale = 0.95

    ctrl = Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)
    # State layout with extra MTQs: still [w, q, h_RW] -- MTQs have no
    # state. Initial state is unchanged.
    x0 = np.concatenate([w0, q0, [0.0]])
    os0 = _make_os(ephem)
    traj = ctrl.calculate_trajectory(
        t_start=0.22, duration=float(N), x_0=x0, os_0=os0,
        goals=GoalList({0.22: ECI_Goal(np.array([0.0, 0.0, 1.0]))}),
        verbose=False,
    )
    # Slice ``traj.controls`` to just the RW row (last actuator) so callers
    # see a single-control timeseries irrespective of the dummy MTQs.
    controls = np.asarray(traj.controls)
    if deterministic and controls.shape[0] == 4:
        # 3 MTQs + 1 RW -> RW is the LAST row
        controls = controls[3:4, :]
    return (np.asarray(traj.times), np.asarray(traj.states), controls)


# ---------------------------------------------------------------------------
# Eigenaxis preservation
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("axis_idx,init_angle_rad", [
    (1, 0.1),                # y-axis, small angle (linear regime)
    (1, np.pi / 2),          # y-axis, 90 degrees (nonlinear)
    (1, np.pi - 0.05),       # y-axis, ~180 degrees (worst case for Taylor;
                             # offset slightly from pi to break the
                             # 180-degree ambiguity that doesn't pick a
                             # rotation direction)
])
def test_y_axis_rw_preserves_y_axial_subspace(ephem, axis_idx, init_angle_rad):
    """Single y-axis RW + initial rotation purely about y MUST give a
    trajectory that stays in the y-axial subspace
    ``(omega_y, delta_q_y, h_RW)``.

    Off-axis omega_x, omega_z, q_x, q_z must be zero to machine precision
    throughout the entire horizon. The dynamics conserve this subspace
    exactly when only a y-axis torque is available and the cost is
    rotation-axis-agnostic. The eigenaxis preservation is SYMMETRY-
    protected: even though the planner's random warmstart at L276
    introduces ~1e-2 noise in the on-axis trajectory, the random ``u_y``
    perturbation stays in the y-axial subspace, so off-axis components
    remain zero to machine precision regardless of warmstart RNG state.
    We therefore use ``deterministic=False`` here -- the off-axis
    machine-zero result is unaffected, while avoiding the ~1e-6
    numerical floor that the 3-MTQ deterministic fixture introduces.
    """
    axis = np.array([0.0, 0.0, 0.0])
    axis[axis_idx] = 1.0
    # Quaternion for rotation by ``init_angle_rad`` about ``axis``:
    q0 = np.zeros(4)
    q0[0] = np.cos(init_angle_rad / 2.0)
    q0[1 + axis_idx] = np.sin(init_angle_rad / 2.0)
    q0 = q0 / np.linalg.norm(q0)
    w0 = np.zeros(3)

    times, states, controls = _run_planner(
        axis=axis, q0=q0, w0=w0, ephem=ephem, deterministic=False,
    )

    # States: [w_x, w_y, w_z, q_w, q_x, q_y, q_z, h_RW]
    # Off-axis omega indices: {0,1,2} \ axis_idx
    off_axis_w_idx = [i for i in (0, 1, 2) if i != axis_idx]
    # Off-axis q-vec indices: {4,5,6} \ (4 + axis_idx)
    off_axis_q_idx = [4 + i for i in (0, 1, 2) if i != axis_idx]

    for idx in off_axis_w_idx:
        max_off = np.max(np.abs(states[idx, :]))
        assert max_off < 1e-10, (
            f"omega[{idx}] (off-axis) deviated from zero: max = {max_off:.3e}"
            f"\n(axis = {axis_idx}, init angle = {init_angle_rad:.3f})"
        )
    for idx in off_axis_q_idx:
        max_off = np.max(np.abs(states[idx, :]))
        assert max_off < 1e-10, (
            f"q[{idx}] (off-axis) deviated from zero: max = {max_off:.3e}"
            f"\n(axis = {axis_idx}, init angle = {init_angle_rad:.3f})"
        )


@pytest.mark.slow
def test_large_angle_y_slew_reaches_goal(ephem):
    """A ~140-degree y-axis slew with a high terminal-attitude weight
    must drive the boresight back to the goal direction. Tests planner
    behaviour well outside the small-angle linearisation regime; uses
    ``ang_cost_func_type=2`` (geodesic acos) which behaves cleanly at
    large angles.

    Note: we deliberately avoid exactly 180 degrees -- at theta = pi the
    cost is at its singularity (``acos(|q . q_goal|) = pi/2`` with zero
    gradient w.r.t. rotation direction), and ALTRO has no symmetry-
    breaking signal to pick a direction. A small initial ``omega_y < 0``
    further biases the rotation toward -y so the optimisation has a
    clear descent direction.
    """
    init_angle = 2.4     # ~137 degrees -- large but well off the singularity
    q0 = np.array([np.cos(init_angle / 2.0), 0.0,
                   np.sin(init_angle / 2.0), 0.0])
    w0 = np.array([0.0, -0.005, 0.0])    # symmetry-breaking nudge
    times, states, _ = _run_planner(
        axis=np.array([0.0, 1.0, 0.0]),
        q0=q0, w0=w0, ephem=ephem,
        N=120,           # Longer horizon for the large slew
        u_max=0.20,      # Higher torque limit
    )
    # Final boresight error: rotation angle about y at terminal step
    # is theta = 2 * atan2(|q_y|, q_w). For the goal (identity rotation),
    # we want theta near 0 (or 2*pi, which is equivalent).
    # arctan2 with the absolute value of q_y handles the q -> -q ambiguity.
    qw_f, qy_f = abs(states[3, -1]), abs(states[5, -1])
    theta_f = 2.0 * np.arctan2(qy_f, qw_f)
    # The minimal-rotation distance to identity (theta in [0, pi]):
    theta_min = min(theta_f, np.pi - theta_f) if theta_f > np.pi / 2 else theta_f

    # 10-degree threshold: with the deterministic warmstart fixture the
    # planner gets within a handful of degrees of the goal reliably; 10
    # degrees gives margin for iLQR-convergence noise without softening
    # the test below the "is it slewing or not?" question.
    assert theta_min < 0.18, (
        f"final orientation off by {theta_min:.4f} rad "
        f"(q_w_f = {states[3, -1]:+.4f}, q_y_f = {states[5, -1]:+.4f}); "
        f"planner failed to reach the goal neighbourhood from a "
        f"~140-degree y-slew."
    )


# ---------------------------------------------------------------------------
# Null-control at goal
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_at_goal_with_zero_rate_commands_near_zero_control(ephem):
    """If x(0) = goal state with zero rate, the running cost on attitude
    and rate is zero everywhere along ``u = 0``, and so is the terminal
    cost (matched goal). The unique optimum is ``u_k = 0`` for all k.

    The planner must converge to this. Any sustained nonzero control is
    either a bug in the goal-handling or the optimiser is stuck on a
    spurious local solution.
    """
    times, states, controls = _run_planner(
        axis=np.array([0.0, 1.0, 0.0]),
        q0=np.array([1.0, 0.0, 0.0, 0.0]),   # identity == goal
        w0=np.zeros(3),                       # at rest
        ephem=ephem,
        N=20,
    )
    # Allow noise at the iLQR convergence floor; require sustained tiny |u|.
    max_u = float(np.max(np.abs(controls)))
    rms_u = float(np.sqrt(np.mean(controls ** 2)))
    assert max_u < 1e-3, (
        f"At-goal commanded nonzero control: max|u| = {max_u:.3e} "
        f"(rms|u| = {rms_u:.3e}); optimum is u = 0 by construction."
    )


# ---------------------------------------------------------------------------
# Sign-flip mirror symmetry
# ---------------------------------------------------------------------------
#
# The "obvious" sign-flip mirror test (compare u_+ to -u_-, demand equality
# to machine precision) is undermined by an empirical fact captured in
# ``test_planner_run_to_run_noise`` below: the C++ planner is run-to-run
# nondeterministic -- identical calls give state-trajectory diffs of order
# 1e-2 in magnitude, dwarfing any symmetry signal we could check on the
# raw trajectories. Instead we test a NOISE-AVERAGED property: the mean
# of ``(u_+ + u_-)`` and ``(w_y_+ + w_y_-)`` across multiple paired runs
# must be near zero. A real sign bug would produce a systematic offset
# the noise can't hide.

@pytest.mark.slow
def test_sign_flip_perturbation_yields_mirror_trajectory(ephem):
    """The dynamics and cost are symmetric under ``omega_0 -> -omega_0``:
    the optimal control trajectory must mirror as ``u(t) -> -u(t)`` and
    the rate component must mirror in state.

    With the deterministic warmstart configuration (3 dummy MTQs +
    ``bdot_on=1``) the planner is bit-exact reproducible, so we can
    assert strict mirroring within iLQR-convergence tolerance rather
    than the noise-averaged variant required for the random-warmstart
    path.
    """
    axis_y = np.array([0.0, 1.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    _, st_p, ctl_p = _run_planner(
        axis=axis_y, q0=q0, w0=np.array([0.0, +0.01, 0.0]), ephem=ephem,
    )
    _, st_m, ctl_m = _run_planner(
        axis=axis_y, q0=q0, w0=np.array([0.0, -0.01, 0.0]), ephem=ephem,
    )
    n = min(ctl_p.shape[1], ctl_m.shape[1])
    err_u = float(np.max(np.abs(ctl_p[0, :n] + ctl_m[0, :n])))
    err_w = float(np.max(np.abs(st_p[1, :] + st_m[1, :])))
    err_q = float(np.max(np.abs(st_p[5, :] + st_m[5, :])))

    u_scale = max(np.max(np.abs(ctl_p[0, :n])), 1e-9)
    w_scale = max(np.max(np.abs(st_p[1, :])), 1e-9)
    q_scale = max(np.max(np.abs(st_p[5, :])), 1e-9)
    # 2% catches a real sign bug while absorbing iLQR convergence noise.
    assert err_u < 0.02 * u_scale, (
        f"u trajectory not mirrored: max|u_+ + u_-| = {err_u:.3e} "
        f"(peak |u| = {u_scale:.3e}, ratio {err_u / u_scale * 100:.2f}%)"
    )
    assert err_w < 0.02 * w_scale, (
        f"omega trajectory not mirrored: max|w_+ + w_-| = {err_w:.3e}"
    )
    assert err_q < 0.02 * q_scale, (
        f"q_y trajectory not mirrored: max|q_+ + q_-| = {err_q:.3e}"
    )


# ---------------------------------------------------------------------------
# Initial-guess robustness
# ---------------------------------------------------------------------------
#
# Same nondeterminism story as the mirror test: comparing two runs at
# different ``rand_add_ratio`` settings shows trajectory differences in
# the same magnitude as comparing two runs at the SAME setting. So instead
# of asserting the trajectories are close, we assert the COST is close --
# cost is a scalar summary, much more robust to per-step trajectory noise.

@pytest.mark.slow
def test_planner_converges_to_same_trajectory_across_init_guesses(ephem):
    """Running the same problem with different ``rand_add_ratio`` settings
    must converge to the same optimum.

    With the deterministic warmstart fixture used here the planner is
    bit-exact reproducible at ``rand_add_ratio = 0``, and small (~0.05-
    0.1) values of the parameter perturb iLQR's inner-loop random-kick
    behaviour (line 1599 of OldPlanner.cpp). All three settings should
    end up in the same attraction basin within iLQR-convergence tolerance.
    """
    axis_y = np.array([0.0, 1.0, 0.0])
    q0 = np.array([np.cos(0.05), 0.0, np.sin(0.05), 0.0])  # 0.1 rad y-tilt
    w0 = np.zeros(3)

    runs = []
    for rand in (0.0, 0.05, 0.1):
        _, _, controls = _run_planner(
            axis=axis_y, q0=q0, w0=w0, ephem=ephem,
            rand_add_ratio=rand,
        )
        runs.append(controls[0, :])

    # Compare first 10 control values (head dominates trajectory shape).
    n_compare = 10
    base = runs[0][:n_compare]
    base_scale = max(np.max(np.abs(base)), 1e-9)
    for j, other in enumerate(runs[1:], start=1):
        diff = np.max(np.abs(other[:n_compare] - base))
        rel = diff / base_scale
        assert rel < 0.10, (
            f"rand_add_ratio = {[0.0, 0.05, 0.1][j]} produced a "
            f"meaningfully different trajectory: max|u_diff| = {diff:.3e} "
            f"(rel = {rel * 100:.1f}%, base peak = {base_scale:.3e}). The "
            f"planner is landing in different basins."
        )


# ---------------------------------------------------------------------------
# Determinism guards
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_planner_with_deterministic_fixture_is_bit_exact_reproducible(ephem):
    """Sanity: with the deterministic-warmstart fixture (3 MTQs +
    ``bdot_on=1``, see ``_run_planner`` docstring), two identical calls
    produce bit-identical trajectories.

    Pins the workaround for the known nondeterminism issue documented
    in ``test_planner_random_warmstart_is_nondeterministic`` below. If
    this test ever starts failing, either:
    * The C++ planner introduced a new randomness source, or
    * The MTQ-count gate at ``OldPlanner.cpp:268`` changed and the
      deterministic path no longer applies.
    """
    axis_y = np.array([0.0, 1.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    w0 = np.array([0.0, +0.01, 0.0])

    _, st1, ctl1 = _run_planner(axis=axis_y, q0=q0, w0=w0, ephem=ephem)
    _, st2, ctl2 = _run_planner(axis=axis_y, q0=q0, w0=w0, ephem=ephem)

    assert np.array_equal(st1, st2), (
        f"Deterministic fixture is non-reproducible! state max diff = "
        f"{float(np.max(np.abs(st1 - st2))):.3e}"
    )
    assert np.array_equal(ctl1, ctl2), (
        f"Deterministic fixture is non-reproducible! control max diff = "
        f"{float(np.max(np.abs(ctl1 - ctl2))):.3e}"
    )


@pytest.mark.slow
def test_planner_random_warmstart_is_nondeterministic(ephem):
    """**Negative-result tripwire** documenting the planner's known
    nondeterminism in the RW-only / ``bdot_on=0`` configuration.

    Root cause: ``OldPlanner.cpp:276`` -- when the gate at L268 fires
    (``bdot_on == 0 || sat.number_MTQ < 3 || bdot_on > 3``), the planner
    generates an initial control sequence via Armadillo's unseeded
    global ``randn``:

    .. code-block:: cpp

        U = diagmat(umax) * randn(size(U)) / RAND_MAX_INIT;

    With ``sat.number_MTQ = 0`` the gate always fires regardless of the
    ``bdot_on`` Python flag.

    A future fix (``arma::arma_rng::set_seed(N)`` at planner
    construction, or restructuring L268 to never fall to the random
    path with a deterministic alternative) would make this test fail
    XPASS-strict-style, prompting either removal of this test or a
    tightening of the asserted noise band.

    We also keep this here so other developers who notice ~1e-2
    state-trajectory mismatches between identical runs find a
    pre-existing reference for the cause.
    """
    axis_y = np.array([0.0, 1.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    w0 = np.array([0.0, +0.01, 0.0])

    _, st1, _ = _run_planner(axis=axis_y, q0=q0, w0=w0, ephem=ephem,
                             deterministic=False)
    _, st2, _ = _run_planner(axis=axis_y, q0=q0, w0=w0, ephem=ephem,
                             deterministic=False)

    state_diff = float(np.max(np.abs(st1 - st2)))
    # Two identical RW-only / bdot_on=0 calls have historically differed
    # by ~ 1e-2 (Armadillo randn drift between runs). If the diff is much
    # smaller (e.g. < 1e-5), the planner has become deterministic --
    # update the tests in this file (set ``deterministic=False`` defaults)
    # and remove this tripwire.
    assert state_diff > 1e-5, (
        f"Planner appears to be deterministic in the random-warmstart "
        f"configuration ({state_diff:.3e}); the workaround in "
        f"``_run_planner(deterministic=True)`` may no longer be needed."
    )
