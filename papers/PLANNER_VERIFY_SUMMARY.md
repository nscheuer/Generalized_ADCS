# Planner paper — verification + new-runs summary (waves A–D, 2026-06-05)

Companion to the per-item files (all in `Paper2_Planner/`). Planner paper only.

## A — claim verification (`VERIFY_A_planner.md`)
- **A1 🔴 paper wrong:** the cold-gas demo replans **every 500 s** (600 s horizon, schedule
  t=0/500/1000/1500); the "1 s" is the *integration step*, not the replan interval. No disturbance
  trigger in the cold-gas case.
- **A2:** spin (IV-C) is an **open-loop PLAN** (label it as such); **twin/mismatch (IV-F) is NOT
  Monte-Carlo** (15 deterministic single-trajectory sweeps).
- **A3 ✅:** the 3-term split is grounded — optimizer "converged" = **constraint satisfaction**, not
  goal tolerance; dynamic feasibility holds by rollout regardless; the paper's <5° is a separate
  post-hoc metric.
- **A4 ✅:** all multi-goal goals are reduced-attitude ECI vectors.
- **A5 ✅:** the LP baseline uses the real **sensed IGRF-13** field (not a dipole).
- **A6 🟡:** `\needval{}` still unfilled (J=diag(0.0314,0.0341,0.0100), MTQ 0.2, RW 2.3/3.6 mN·m·s,
  dt 1 s); orbit is "radius" (correct); orbit randomization is equivalent to uniform incl/RAAN/phase.
- **A7 ✅:** N=100/cell matrix, 100 multi-goal, 6/setting hyperparam, 15 twin sweeps.
- **A8 ✅:** MPC tested, underperformed (TVLQR 100%/0.12° vs MPC 62%/13°) — empirical only.
- **A9 ✅ (exact):** from the committed `autonomous_spin.py` (SALTRO `autonomous-spin-backup`, Table 7.3):
  prop τ_d = 0.3 mN·m on +x̂, RW on +ŷ, h_max = 2 mN·m·s ⇒ wrong-axis + **t_sat = 2/0.3 = 6.7 s**. It's
  `trajOpt` (open-loop plan), confirming A2. The local `generate_fig_spin.py` is stale — don't quote it.

## B — recompute / new runs
- **B1 (`MULTIGOAL_HELD_RESULTS.md`, `tab_multigoal_redef.*`):** the old "held = final-in-window <5°"
  **undersells** the planner (it anticipatorily starts the next slew). Redefined **held = ≥100 s
  continuous <5°** (recomputed from saved `.sim`): goal A/B go from **10–25% → 77–100%**, with
  time-to-acquire and steady-state error (0.17–1.28°). Use this table.
- **B2 (`fig_environment_sweep.*`, `ENVIRONMENT_SWEEP_RESULTS.md`):** replaces the hard-to-read CDF —
  **100% convergence across all 6 environments** {none, dipole-0.05/0.10, GG+drag+SRP, full-0.05/0.10};
  mean final error scales with dipole (0.06°→0.80°), aero alone negligible. All disturbances excluded
  from the planner model.
- **B3:** `fig_itworks_planner_vs_pd` right panel now **linear** (planner bars hug zero vs tall PD bars).

## C — new demos (effort-first)
- **C1 ✅ (`fig_c1_boresight_roll.*`, `C1_RESULTS.md`):** **foresight pre-positioning demonstrated** —
  with a single RW and vector goals, the planner rolls about the free boresight during the A-hold to
  swing the wheel axis 0.03→0.63 toward the upcoming slew axis (pre-loading the wheel), pointing
  undisturbed. Backs the abstract's foresight claim.
- **C2 ❌ honest negative (`fig_c2_dipole_desat.*`, `C2_RESULTS.md`):** GG/aero are physically too weak
  to desaturate (6.9%/0.8% of h_max per orbit). A strong modeled dipole (0.2 A·m²) **diverged** on the
  OldPlanner (wheel→63 mN·m·s, pointing lost) — it has no momentum-minimization cost. A real demo needs
  the momentum-aware SALTRO planner + a moderate disturbance (deferred, effort HIGH).

## D — adversarial audit (`WAVE_D_audit.md`)
- **FINDING 1 [MED-HIGH]:** the twin **field**-mismatch "looks too good" because the **tracker senses
  the perturbed field** (`find_u` gets the true `os_k`) — so B-mag/rotation mismatch only corrupts the
  open-loop plan. Reframe as "a nominal-field plan stays trackable by a field-*sensing* tracker"; lead
  the twin story with the **J-mismatch** (J is not sensed — genuine).
- **FINDING 2 [LOW]:** the multi-goal "representative" figure is the **best-case** trial — label it.
- **CLEAN:** pairing (all `base_seed=42`), no filtering/cherry-picking, no future-info leakage, planner
  genuinely blind to plant disturbances/true inertia. The mechanics are honest.

## Bonus (from earlier this session)
- `fig_disturbance_robustness.*` / `DISTURBANCE_ROBUSTNESS_RESULTS.md` — backs the abstract's disturbance
  clause (100%→100% conv under unmodeled 1.1 µN·m dipole + drag/SRP + GG).
- `fig_mtq_failure_modes.*` / `MTQ_16PCT_FAILURE_NOTE.md` — the 3+0 reduced 16% non-convergence is
  horizon-truncation (5) + terminal MTQ-hold limit (11), not the slew-blocking first guessed.
