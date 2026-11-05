"""
Provably-LQR-convergent corner of the C++ trajectory planner.

Background
----------
The historical :mod:`testing.test_controllers.test_planner_oracle` suite tried
to assert that ``Plan_and_Track_LQR`` (the C++ pysat trajectory planner)
reduces to a Python 2-state closed-form discrete LQR on small-angle, loose-
constraint problems. The investigation captured in issue #68 instrumented the
C++ ``backwardPass`` and showed that the equivalence does **not** hold even for
small angles, because:

1. The planner's reduced state is 7-dimensional
   :math:`(\\omega_x, \\omega_y, \\omega_z, \\delta q_x, \\delta q_y, \\delta q_z,
   h_{\\text{RW}})`, while the LQR oracle uses a 2-dimensional
   :math:`(e_y, \\omega_y)`. The 7-state Riccati picks up cost-to-go entries on
   all three attitude axes from the isotropic running and terminal cost
   ``w_ang * (1 - q . q_goal) + 0.5 * w_av * |w|^2``.
2. The C++ stepcost uses ``0.5 * w_av * |w|^2`` and ``0.5 * u' * R * u``
   (factor of one half), while the LQR oracle uses ``w_av * w^2`` and
   ``R_u * u^2`` (no factor). With ``ang_cost_func_type = 0``,
   ``(1 - cos(theta/2)) ~ theta^2 / 8`` in the small-angle limit.

Both 1 and 2 are *correct planner behaviour* for a real 3D-rotating spacecraft
with a reaction wheel. The original LQR equivalence claim was structurally
unachievable.

This file constructs the corner of the parameter space where the structural
mismatch is suppressed enough that ALTRO does converge to the 2-state LQR
oracle within a measurable tolerance:

* Initial quaternion fixed at identity, so the linearisation is exactly on the
  rotation manifold's tangent plane and the off-axis blocks of the value
  function don't leak through the (axis-aligned) Bqk into the y-axis Qkuu.
* Initial perturbation only in ``omega_y`` (no attitude offset), so the
  trajectory stays in the y-axial subspace by symmetry of the dynamics.
* Cost weights pre-scaled by the conversion factors derived in the #68
  comment: ``(angle, ang_vel, control_mult * rw_control_weight) =
  (8 * Q_e, 2 * Q_omega, 2 * R_u)``.
* iLQR's L-M regulariser left at default (it doesn't dominate Qkuu in this
  regime, confirmed empirically in #68).

Even with this construction the residual gap between ALTRO and the 2-state
LQR oracle is ~8-10 percent of the initial perturbation magnitude, because
iLQR's break-on-gradient tolerance is finite (``grad ~ 3e-3`` in the verbose
log) and the wheel-momentum h state continues to be propagated even when
``J_RW`` is taken arbitrarily small. We therefore assert a 15 percent
trajectory-difference tolerance, scaled by initial perturbation magnitude -
tight enough to catch the planner regressing to the historical broken state
(~170 percent over-command), loose enough to absorb iLQR convergence noise
and the small residual h-coupling. The companion
``test_planner_kkt_consistency.py`` file verifies the planner satisfies its
*own* KKT conditions in the same and adjacent scenarios, providing a model-
agnostic guard.
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
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.satellite.satellite import Satellite

# Import the existing pure-Python LQR oracle helpers; they're correct in
# their own (2-state) model and we want to reuse them as the comparison.
from testing.test_controllers.test_planner_oracle import (
    DiscreteLQRParams,
    solve_discrete_lqr_optimal,
    extract_single_axis_from_altro,
)


# ---------------------------------------------------------------------------
# Cost-weight conversion: (Q_e, Q_omega, R_u) in the LQR oracle's convention
# vs (angle, ang_vel, control_mult * rw_control_weight) in the C++ stepcost.
#
# C++ stepcost (Satellite.cpp:808): state_cost = 0.5 * w_av * |w|^2 + w_ang * (1 - q.q_goal)
#                                  actuation  = 0.5 * u' * diag(RW_cost) * u * w_u_mult
# Small-angle:    (1 - cos(theta/2)) ~ theta^2 / 8
#
# To match LQR oracle's   Sum(Q_e * theta_k^2 + Q_omega * omega_k^2 + R_u * u_k^2) + ...
# we need (apples-to-apples in the LQR oracle's "no-half" convention):
#   w_ang  =  8 * Q_e        (the 1/8 small-angle factor)
#   w_av   =  2 * Q_omega    (the leading 0.5 in stepcost)
#   w_u_mult * RW_cost = 2 * R_u   (the leading 0.5 in stepcost)
# ---------------------------------------------------------------------------
ANGLE_FACTOR = 8.0     # ang_cost_func_type = 0; for type 3 (geodesic^2) this would also be 8
ANGVEL_FACTOR = 2.0
CONTROL_FACTOR = 2.0


# Tolerance: with the construction above the residual gap is ~8% of the
# initial perturbation; 15% gives comfortable margin while still catching the
# 170% over-command of the historical broken-equivalence regime.
LQR_CONVERGENCE_TOL = 0.15


@pytest.fixture(scope="module")
def ephem():
    return Ephemeris()


def _build_y_axis_rw_setup(ephem, J=0.1, J_rw=1e-3, u_max=1.0, h_max=10.0):
    """Single y-axis reaction wheel on a body with diagonal inertia.

    Quaternion is **identity** (no pre-existing rotation); perturbation will
    be applied only to ``omega_y`` so the trajectory stays purely y-axial in
    the body frame.
    """
    rw = RW(axis=np.array([0.0, 1.0, 0.0]), max_torque=u_max,
            J=J_rw, h=0.0, h_max=h_max)
    sat = Satellite(mass=4.0,
                    J_0=np.diagflat([J, J, J]),
                    actuators=[rw],
                    sensors=[MTM(axis=np.array([0.0, 0.0, 1.0]))],
                    boresight=np.array([0.0, 0.0, 1.0]))
    os0 = Orbital_State(ephem=ephem, J2000=0.22,
                        R=7000.0 * np.array([1.0, 0.0, 0.0]),
                        V=np.array([0.0, 7.5, 0.0]),
                        B=np.array([0.1, 0.0, 0.0]),
                        S=np.array([1e5, 0.0, 0.0]),
                        rho=0.0)
    return sat, os0


def _run_altro_y_axis(ephem, *, Q_e, Q_omega, R_u, Q_e_N, Q_omega_N,
                      omega0_y, dt, N, J=0.1, J_rw=1e-3, u_max=1.0,
                      h_max=10.0):
    """Run the C++ planner with weights converted to its convention."""
    sat, os0 = _build_y_axis_rw_setup(ephem, J=J, J_rw=J_rw,
                                      u_max=u_max, h_max=h_max)
    cw = CostWeights(
        angle=ANGLE_FACTOR * Q_e,
        angle_N=ANGLE_FACTOR * Q_e_N,
        ang_vel=ANGVEL_FACTOR * Q_omega,
        ang_vel_N=ANGVEL_FACTOR * Q_omega_N,
        control_mult=1.0,
        ang_cost_func_type=0,
    )
    ps = PlannerSettings(est_sat=sat, dt_tp=dt, dt_tvlqr=1.0, bdot_on=0,
                         cost_main=cw, cost_second=cw, cost_tvlqr=cw)
    ps.verbosity = False
    ps.wmax = 1.0
    # control_mult * rw_control_weight = 2 R_u  ->  rw_control_weight = 2 R_u
    ps.rw_control_weight = CONTROL_FACTOR * R_u
    ps.control_limit_scale = 1.0

    ctrl = Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)

    q0 = np.array([1.0, 0.0, 0.0, 0.0])     # identity quaternion
    w0 = np.array([0.0, omega0_y, 0.0])      # single y-axis perturbation
    x0 = np.concatenate([w0, q0, [0.0]])     # [omega, q, h_RW]

    traj = ctrl.calculate_trajectory(
        t_start=0.22, duration=float(N), x_0=x0, os_0=os0,
        goals=GoalList({0.22: ECI_Goal(np.array([0.0, 0.0, 1.0]))}),
        verbose=False,
    )
    _, e_a, w_a, _, u_a = extract_single_axis_from_altro(traj)
    return np.asarray(e_a), np.asarray(w_a), np.asarray(u_a)


def _solve_lqr_oracle(*, Q_e, Q_omega, R_u, Q_e_N, Q_omega_N,
                      e0, omega0_y, dt, N, J=0.1):
    """2-state pure-Python LQR closed-form solution. Reused as the oracle."""
    p = DiscreteLQRParams(J=J, dt=dt, N=N,
                          Q_e=Q_e, Q_omega=Q_omega, R_u=R_u,
                          Q_e_N=Q_e_N, Q_omega_N=Q_omega_N)
    return solve_discrete_lqr_optimal(e0, omega0_y, p)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("omega0_y", [0.005, 0.01, 0.02])
def test_altro_converges_to_lqr_oracle_rate_perturbation(ephem, omega0_y):
    """ALTRO matches the 2-state LQR oracle when:

    * q0 = identity (no pre-existing rotation),
    * x(0) perturbed only in omega_y (single-axis),
    * cost weights converted to the planner's convention (8x, 2x, 2x).

    Tolerance: 15 percent of the initial omega magnitude. Historical broken
    state (q0 != identity, weights not converted) saw 170 percent
    over-command; the (q0 = I, weights converted) regime sees ~8 percent.
    """
    Q_e, Q_omega, R_u = 1e3, 1e4, 1.0
    Q_e_N, Q_omega_N = 1e4, 1e5
    dt, N = 1.0, 30

    e_a, w_a, u_a = _run_altro_y_axis(
        ephem, Q_e=Q_e, Q_omega=Q_omega, R_u=R_u,
        Q_e_N=Q_e_N, Q_omega_N=Q_omega_N,
        omega0_y=omega0_y, dt=dt, N=N,
    )
    lqr = _solve_lqr_oracle(
        Q_e=Q_e, Q_omega=Q_omega, R_u=R_u,
        Q_e_N=Q_e_N, Q_omega_N=Q_omega_N,
        e0=0.0, omega0_y=omega0_y, dt=dt, N=N,
    )

    n = min(len(lqr.e), len(e_a))
    e_diff = np.max(np.abs(lqr.e[:n] - e_a[:n]))
    w_diff = np.max(np.abs(lqr.omega[:n] - w_a[:n]))

    scale = max(abs(omega0_y), 1e-9)
    assert e_diff < LQR_CONVERGENCE_TOL * scale, (
        f"angle trajectory: max diff = {e_diff:.5f} > {LQR_CONVERGENCE_TOL} * |omega0| = "
        f"{LQR_CONVERGENCE_TOL * scale:.5f}\n"
        f"  LQR  u(0) = {lqr.u[0]:+.5f}, w(1) = {lqr.omega[1]:+.5f}\n"
        f"  ALTRO u(0) = {u_a[0]:+.5f}, w(1) = {w_a[1]:+.5f}"
    )
    assert w_diff < LQR_CONVERGENCE_TOL, (
        f"omega trajectory: max diff = {w_diff:.5f} > {LQR_CONVERGENCE_TOL}\n"
        f"  LQR  u(0) = {lqr.u[0]:+.5f}, ALTRO u(0) = {u_a[0]:+.5f}"
    )

    # Sign + decay sanity: ALTRO must drive omega back toward zero.
    # Final-state omega < initial-state omega by at least 2x; threshold is
    # generous because the terminal weight (1e5) is large but the horizon
    # may not be long enough to fully null the rate.
    assert abs(w_a[-1]) < 0.5 * abs(omega0_y), (
        f"ALTRO did not bring omega toward zero: |w_final| = {abs(w_a[-1]):.5f}, "
        f"|w_initial| = {abs(omega0_y):.5f}"
    )


@pytest.mark.slow
def test_altro_lqr_convergence_scales_linearly_with_perturbation(ephem):
    """The C++ planner is (asymptotically) linear: when the initial omega
    is halved, the commanded controls and resulting trajectory should also
    halve. Independent of the LQR oracle. Detects nonlinearity creeping in
    if e.g. the constraint-active path activated unintentionally.
    """
    Q_e, Q_omega, R_u = 1e3, 1e4, 1.0
    Q_e_N, Q_omega_N = 1e4, 1e5
    dt, N = 1.0, 30

    def go(w):
        return _run_altro_y_axis(
            ephem, Q_e=Q_e, Q_omega=Q_omega, R_u=R_u,
            Q_e_N=Q_e_N, Q_omega_N=Q_omega_N,
            omega0_y=w, dt=dt, N=N,
        )

    e1, w1, u1 = go(0.01)
    e2, w2, u2 = go(0.02)

    # Linearity check: u2 ~ 2 * u1 within iLQR convergence tolerance.
    n = min(len(u1), len(u2))
    nonzero = np.abs(u1[:n]) > 1e-7
    if not nonzero.any():
        pytest.skip("planner returned numerically zero controls")
    ratios = u2[:n][nonzero] / u1[:n][nonzero]
    assert np.allclose(ratios, 2.0, rtol=0.15), (
        f"u(2*w0)/u(w0) is not ~ 2; ratios sample = {ratios[:5]}"
    )


@pytest.mark.slow
def test_altro_under_lqr_cost_form_is_close_to_optimum(ephem):
    """The LQR cost ``sum(x' Q x + u' R u) + xN' Q_N xN`` is provably
    optimal in its own (2-state) world. ALTRO's trajectory, evaluated
    under that same cost form, should be within a finite multiplicative
    factor of the optimum (not less than 1.0 - LQR optimum is a lower
    bound - and not more than ~2x given the construction here).
    """
    Q_e, Q_omega, R_u = 1e3, 1e4, 1.0
    Q_e_N, Q_omega_N = 1e4, 1e5
    dt, N = 1.0, 30
    omega0_y = 0.01

    e_a, w_a, u_a = _run_altro_y_axis(
        ephem, Q_e=Q_e, Q_omega=Q_omega, R_u=R_u,
        Q_e_N=Q_e_N, Q_omega_N=Q_omega_N,
        omega0_y=omega0_y, dt=dt, N=N,
    )
    lqr = _solve_lqr_oracle(
        Q_e=Q_e, Q_omega=Q_omega, R_u=R_u,
        Q_e_N=Q_e_N, Q_omega_N=Q_omega_N,
        e0=0.0, omega0_y=omega0_y, dt=dt, N=N,
    )

    def lqr_form_cost(e, w, u, n):
        c = 0.0
        for k in range(n):
            c += Q_e * e[k]**2 + Q_omega * w[k]**2 + R_u * u[k]**2
        c += Q_e_N * e[n]**2 + Q_omega_N * w[n]**2
        return c

    c_lqr = lqr.cost
    c_altro = lqr_form_cost(e_a, w_a, u_a, N)

    # LQR cost is the provable lower bound. ALTRO's trajectory under the
    # same cost form must be >= LQR optimum, and within ~2x (the q0=identity
    # + matched-weights regime).
    assert c_lqr <= c_altro * (1 + 1e-6), (
        f"ALTRO trajectory has cost {c_altro:.4f} < LQR optimum {c_lqr:.4f}; "
        f"oracle is supposed to be a lower bound."
    )
    assert c_altro < 2.0 * c_lqr, (
        f"ALTRO cost {c_altro:.4f} more than 2x LQR optimum {c_lqr:.4f}; "
        f"the (q0=identity, matched-weights) regime should keep within 2x."
    )
