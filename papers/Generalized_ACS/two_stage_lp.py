"""
Two-stage (lexicographic) LP torque allocator  —  Task 2a refinement.

THE DEFECT (see EXEC_BRIEF2_RESULTS.md section 2a, demo_lp_precession_failure.py)
-------------------------------------------------------------------------------
The controller hands the allocator a single FUSED torque request

    tau_des = tau_pd + tau_gyro

where tau_pd is the discretionary pointing feedback and
tau_gyro = w x (J w + h_rw) is the NON-NEGOTIABLE gyroscopic-compensation
feedforward (you must apply exactly tau_gyro just to hold the current rate).
The single-stage direction-preserving LP maximizes one scalar alpha and, when
saturated, returns alpha*(tau_pd + tau_gyro) -- it under-delivers BOTH terms
by the same fraction, including the mandatory tau_gyro. At high stored RW
momentum alpha collapses to ~0.03.

THE FIX (two-stage / lexicographic LP)
--------------------------------------
Stage 1 -- pay tau_gyro first: solve the direction-preserving LP for tau_gyro
           ALONE over the full actuator polytope. If feasible, deliver it
           exactly; if not, deliver the max feasible toward it (and there is
           then nothing left for Stage 2 -- correct: if you cannot even
           cancel precession you should not be spending budget on pointing).
Stage 2 -- spend the LEFTOVER: with the Stage-1 command u_g committed, solve
           the direction-preserving LP for tau_pd using only the remaining
           per-actuator authority, i.e. the additional command u_p must keep
           u_g + u_p inside the box [-u_max, u_max].
Command u = u_g + u_p.

This protects the precession budget (tau_gyro gets first claim on the
polytope) while keeping the LP direction-preservation property for the
pointing torque -- the property RESEARCH_MASTER.md credits LP for.

This module provides:
  * allocate_single_stage_lp(...)  -- the current behaviour, for comparison.
  * allocate_two_stage_lp(...)     -- the refinement.
Both take the SAME matrix inputs and the SAME A-matrix convention as
ADCS.controller.MTQ_w_RW_LP.allocate_max_torque_in_direction
(A_total = [A_rw | -skew(b) A_mtq_axes]).

Run as a script: reproduces the demo_lp_precession_failure.py PART A sweep for
BOTH allocators side by side and shows the gyro shortfall eliminated.

  PYTHONPATH=. venv/bin/python papers/Generalized_ACS/two_stage_lp.py
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np
from scipy.optimize import linprog

from ADCS.helpers.math_helpers import skewsym


# -----------------------------------------------------------------------------
# Core LP primitive — max feasible torque along a direction, with a box on u.
# -----------------------------------------------------------------------------
def _max_torque_along(A_total: np.ndarray, tau: np.ndarray,
                      lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """
    Direction-preserving LP:  max T  s.t.  A_total u = T*tau_hat,  lb<=u<=ub, T>=0.

    Returns (u, T_max, success). u is the maximizing command (NOT yet scaled
    to tau); T_max is the largest achievable magnitude along tau_hat.
    """
    t_mag = float(np.linalg.norm(tau))
    n_act = A_total.shape[1]
    if t_mag < 1e-12:
        return np.zeros(n_act), 0.0, True

    tau_hat = tau / t_mag
    # decision vars x = [u (n_act), T]; maximize T => minimize -T
    c = np.zeros(n_act + 1)
    c[-1] = -1.0
    A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
    b_eq = np.zeros(3)
    bounds = [(lb[i], ub[i]) for i in range(n_act)] + [(0.0, None)]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return np.zeros(n_act), 0.0, False
    return res.x[:n_act], float(res.x[-1]), True


@dataclass
class AllocInfo:
    """Per-call diagnostics."""
    alpha_gyro: float       # fraction of tau_gyro delivered
    alpha_pd: float         # fraction of tau_pd delivered
    gyro_shortfall: float   # (1-alpha_gyro)*||tau_gyro||  [N m]
    saturated_stage1: bool  # True if tau_gyro itself was infeasible


def _build_A(A_rw: np.ndarray, A_mtq_axes: np.ndarray,
             b_body: np.ndarray) -> np.ndarray:
    """A_total = [A_rw | -skew(b) A_mtq_axes] — the controller's convention."""
    if A_mtq_axes.shape[1] > 0:
        A_mtq = -skewsym(b_body) @ A_mtq_axes
    else:
        A_mtq = np.zeros((3, 0))
    return np.hstack([A_rw, A_mtq])


def allocate_single_stage_lp(tau_pd: np.ndarray, tau_gyro: np.ndarray,
                             A_rw: np.ndarray, A_mtq_axes: np.ndarray,
                             b_body: np.ndarray, u_rw_max: np.ndarray,
                             u_mtq_max: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, AllocInfo]:
    """Current behaviour: one LP on the FUSED tau_pd + tau_gyro, uniform alpha."""
    A_total = _build_A(A_rw, A_mtq_axes, b_body)
    n_rw = A_rw.shape[1]
    u_lim = np.concatenate([u_rw_max, u_mtq_max])
    tau_des = tau_pd + tau_gyro
    t_mag = float(np.linalg.norm(tau_des))

    u, T_max, ok = _max_torque_along(A_total, tau_des, -u_lim, u_lim)
    if not ok or t_mag < 1e-12:
        z_rw, z_mtq = np.zeros(n_rw), np.zeros(len(u_mtq_max))
        return z_rw, z_mtq, AllocInfo(1.0, 1.0, 0.0, False)

    if T_max <= t_mag:
        alpha = T_max / t_mag           # saturated: whole vector scaled by alpha
    else:
        u = u * (t_mag / T_max)         # surplus: scale down to match exactly
        alpha = 1.0
    # uniform alpha hits gyro and pd identically
    gyro_norm = float(np.linalg.norm(tau_gyro))
    return (u[:n_rw], u[n_rw:],
            AllocInfo(alpha_gyro=alpha, alpha_pd=alpha,
                      gyro_shortfall=(1.0 - alpha) * gyro_norm,
                      saturated_stage1=(alpha < 1.0 and gyro_norm > 0)))


def _two_stage_ordered(A_total: np.ndarray, u_lim: np.ndarray,
                       primary: np.ndarray, secondary: np.ndarray
                       ) -> tuple[np.ndarray, float, float]:
    """
    Lexicographic two-stage solve for a given priority order.

    Stage 1 delivers ``primary`` over the full polytope (exact if feasible,
    else the max feasible fraction). Stage 2 delivers ``secondary`` using only
    the leftover per-actuator authority. Returns (u, alpha_primary,
    alpha_secondary) -- alpha_* is the delivered fraction of each request.
    """
    n_act = A_total.shape[1]

    pm = float(np.linalg.norm(primary))
    if pm < 1e-12:
        u1, a1 = np.zeros(n_act), 1.0
    else:
        u1, T1, ok1 = _max_torque_along(A_total, primary, -u_lim, u_lim)
        if not ok1:
            u1, a1 = np.zeros(n_act), 0.0
        elif T1 >= pm:
            u1, a1 = u1 * (pm / T1), 1.0          # feasible -> exact
        else:
            a1 = T1 / pm                          # u1 already at max-feasible

    sm = float(np.linalg.norm(secondary))
    if sm < 1e-12:
        u2, a2 = np.zeros(n_act), 1.0
    else:
        u2, T2, ok2 = _max_torque_along(A_total, secondary,
                                        -u_lim - u1, u_lim - u1)
        if not ok2:
            u2, a2 = np.zeros(n_act), 0.0
        elif T2 >= sm:
            u2, a2 = u2 * (sm / T2), 1.0
        else:
            a2 = T2 / sm
    return u1 + u2, a1, a2


def allocate_two_stage_lp(tau_pd: np.ndarray, tau_gyro: np.ndarray,
                          A_rw: np.ndarray, A_mtq_axes: np.ndarray,
                          b_body: np.ndarray, u_rw_max: np.ndarray,
                          u_mtq_max: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, AllocInfo]:
    """
    Two-stage robust LP2 (Nic's LP2, lexicographic gyro-priority form).

    * **Stage 1 — gyro priority, always.** Deliver tau_gyro as fully as the
      polytope allows: exactly if tau_gyro is feasible, otherwise the
      max-feasible point in its direction (the LP1-style alpha applied to
      tau_gyro *alone*). This feasibility fallback is what cures the
      "trivial alpha=0 no longer holds" infeasibility hole Nic flagged in the
      single-LP statement of LP2.

    * **Stage 2 — PD on the remainder, always.** With the Stage-1 command u1
      fixed, maximize the discretionary tau_pd over the *residual* per-actuator
      authority (box ``[-u_lim - u1, u_lim - u1]``). Stage 2 is NEVER skipped:
      even when Stage 1 saturates there is residual authority -- e.g. an
      off-axis tau_gyro leaves the whole RW axis untouched -- and pointing
      should use every bit of it. (Skipping Stage 2 on saturation was a bug:
      it starved pointing to ~0 in the underactuated 3+1 case.)

    Priority is fixed gyro-first; it is NOT flipped when tau_gyro is
    infeasible. The gyro feedforward always gets first claim; PD always gets
    whatever is left.

    Returns (u_rw, u_mtq, info) — same shape/contract as the single-stage form.
    """
    A_total = _build_A(A_rw, A_mtq_axes, b_body)
    n_rw = A_rw.shape[1]
    u_lim = np.concatenate([u_rw_max, u_mtq_max])
    g_mag = float(np.linalg.norm(tau_gyro))

    u, a_gyro, a_pd = _two_stage_ordered(A_total, u_lim, tau_gyro, tau_pd)

    info = AllocInfo(alpha_gyro=a_gyro, alpha_pd=a_pd,
                     gyro_shortfall=(1.0 - a_gyro) * g_mag,
                     saturated_stage1=(a_gyro < 1.0 - 1e-9 and g_mag > 1e-12))
    return u[:n_rw], u[n_rw:], info


# =============================================================================
# Validation. The two-stage LP only changes the outcome when tau_gyro is
# INDIVIDUALLY feasible but the fused tau_gyro + tau_pd is not. Two sweeps:
#
#   Sweep A (3+1): the demo_lp_precession_failure.py operating point. Here
#     tau_gyro is itself infeasible (a single x-RW cannot torque against its
#     own precession; the MTQs are too weak) -- genuine underactuation, NOT a
#     fused-request artifact. Two-stage correctly cannot help. Reported for
#     honesty.
#
#   Sweep B (3+3): tau_gyro is RW-feasible. As the discretionary pointing
#     request tau_pd grows, the fused tau_gyro + tau_pd leaves the polytope.
#     Single-stage scales the whole vector -> gyro under-delivered. Two-stage
#     pays tau_gyro first -> gyro shortfall stays zero, only tau_pd is cut.
#     This is the regime the refinement targets, and where it wins.
# =============================================================================
def _sweep_3p1() -> list:
    """3+1, sweep stored RW momentum (the demo operating point)."""
    J_0 = np.diagflat([0.022, 0.022, 0.004])
    rw_axis = np.array([1.0, 0.0, 0.0])
    A_rw = rw_axis.reshape(3, 1)
    A_mtq_axes = np.eye(3)
    u_rw_max = np.array([7e-3])
    u_mtq_max = np.array([0.4, 0.4, 0.4])
    RW_HMAX = 16.2e-3
    w = np.array([0.0, 0.0, 0.03])
    b_body = np.array([2.0e-5, 1.0e-5, 1.5e-5])
    tau_pd = np.array([1.0e-4, -0.5e-4, 0.3e-4])
    rows = []
    for fr in np.linspace(0.0, 1.0, 21):
        tau_gyro = np.cross(w, J_0 @ w + (fr * RW_HMAX) * rw_axis)
        _, _, i1 = allocate_single_stage_lp(tau_pd, tau_gyro, A_rw, A_mtq_axes,
                                            b_body, u_rw_max, u_mtq_max)
        _, _, i2 = allocate_two_stage_lp(tau_pd, tau_gyro, A_rw, A_mtq_axes,
                                         b_body, u_rw_max, u_mtq_max)
        rows.append((fr, float(np.linalg.norm(tau_gyro)), i1, i2))
    return rows


def _sweep_3p3() -> list:
    """3+3, fixed RW-feasible tau_gyro, sweep the discretionary tau_pd."""
    A_rw = np.eye(3)
    A_mtq_axes = np.eye(3)
    u_rw_max = np.array([7e-3, 7e-3, 7e-3])
    u_mtq_max = np.array([0.4, 0.4, 0.4])
    b_body = np.array([2.0e-5, 1.0e-5, 1.5e-5])
    # tau_gyro fixed, RW-feasible (3e-3 < 7e-3 per-axis box):
    tau_gyro = np.array([0.0, 0.0, 3.0e-3])
    pd_dir = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)  # competes on the z-RW
    rows = []
    for mag in np.linspace(0.0, 1.2e-2, 25):
        tau_pd = mag * pd_dir
        _, _, i1 = allocate_single_stage_lp(tau_pd, tau_gyro, A_rw, A_mtq_axes,
                                            b_body, u_rw_max, u_mtq_max)
        _, _, i2 = allocate_two_stage_lp(tau_pd, tau_gyro, A_rw, A_mtq_axes,
                                         b_body, u_rw_max, u_mtq_max)
        rows.append((mag, float(np.linalg.norm(tau_gyro)), i1, i2))
    return rows


def _validation_sweep() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_a = _sweep_3p1()
    rows_b = _sweep_3p3()

    print("SWEEP A — 3+1, stored-momentum sweep (the demo operating point)")
    print(f"  {'h/h_max':>8} {'|tau_gyro|':>12} {'1-stage gyro short':>20} "
          f"{'2-stage gyro short':>20} {'2-stage a_gyro':>15}")
    for fr, gn, i1, i2 in rows_a:
        print(f"  {fr:8.2f} {gn:12.3e} {i1.gyro_shortfall:20.3e} "
              f"{i2.gyro_shortfall:20.3e} {i2.alpha_gyro:15.3f}")
    a_feasible = [r for r in rows_a if not r[3].saturated_stage1]
    print(f"  -> tau_gyro is individually feasible for only "
          f"{len(a_feasible)}/{len(rows_a)} points (h/h_max <= "
          f"{max((r[0] for r in a_feasible), default=0.0):.2f}). Beyond that the "
          f"3+1 satellite is genuinely underactuated against its own")
    print(f"     wheel precession -- NO allocator can fix it; two-stage "
          f"correctly does not pretend to.")

    print("\nSWEEP B — 3+3, RW-feasible tau_gyro, sweep discretionary tau_pd")
    print(f"  {'|tau_pd|':>12} {'1-stage gyro short':>20} "
          f"{'2-stage gyro short':>20} {'1-stg a_pd':>12} {'2-stg a_pd':>12}")
    for mag, gn, i1, i2 in rows_b:
        print(f"  {mag:12.3e} {i1.gyro_shortfall:20.3e} "
              f"{i2.gyro_shortfall:20.3e} {i1.alpha_pd:12.3f} {i2.alpha_pd:12.3f}")
    w1 = max(r[2].gyro_shortfall for r in rows_b)
    w2 = max(r[3].gyro_shortfall for r in rows_b)
    print(f"  -> worst-case gyro shortfall over the sweep: single-stage "
          f"{w1:.3e} N m  ->  two-stage {w2:.3e} N m")
    if w2 < 1e-9:
        print("     two-stage delivers tau_gyro EXACTLY at every point "
              "(gyro shortfall identically zero) -- the refinement works.")

    # figure — focus on Sweep B (the regime the refinement targets)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    mags = [r[0] * 1e3 for r in rows_b]  # mN m
    axes[0].plot(mags, [r[2].gyro_shortfall * 1e3 for r in rows_b], "o-",
                 color="#de2d26", label="single-stage LP (current)")
    axes[0].plot(mags, [r[3].gyro_shortfall * 1e3 for r in rows_b], "s-",
                 color="#2c7fb8", label="two-stage LP (refined)")
    axes[0].set_xlabel(r"discretionary request  $\|\tau_{pd}\|$  [mN m]")
    axes[0].set_ylabel(r"gyro shortfall  $(1-\alpha_{gyro})\|\tau_{gyro}\|$  [mN m]")
    axes[0].set_title("(a) 3+3: two-stage holds the gyro term exact\n"
                      "while single-stage bleeds it as tau_pd grows")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls="--", alpha=0.4)

    axes[1].plot(mags, [r[2].alpha_pd for r in rows_b], "o-", color="#de2d26",
                 label="single-stage: pointing alpha")
    axes[1].plot(mags, [r[3].alpha_pd for r in rows_b], "s-", color="#2c7fb8",
                 label="two-stage: pointing alpha (leftover)")
    axes[1].set_xlabel(r"discretionary request  $\|\tau_{pd}\|$  [mN m]")
    axes[1].set_ylabel(r"pointing-torque fraction delivered  $\alpha_{pd}$")
    axes[1].set_title("(b) Two-stage trades pointing (not precession)\n"
                      "down under saturation -- the correct priority")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, ls="--", alpha=0.4)

    fig.suptitle("Two-stage LP allocator — validation (3+3: regime the "
                 "refinement targets)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(os.path.dirname(__file__), "output_data",
                       "fig_two_stage_lp_validation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    _validation_sweep()
