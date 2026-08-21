# Freeze proposal: per-task planner configurations (skeleton, pre-validation)

Structure committed BEFORE validation lands; the bracketed values fill in from
TUNE_VALIDATE.txt and the proposal goes to Patrick before the registered rerun.

## FROZEN (assertion-pinned in the rerun generator)

| item | REDUCED (boresight) | FULL (3-axis) | basis |
|---|---|---|---|
| cost_main.angle | 1e2 | 1e2 | sweeps both tasks; 1e3 demoted on full (ceiling duty-cycle: standing h 0.341, 4.4% of orbit > 0.42) |
| cost_main.angle_N | 1e4 | 1e2 (= angle, flat) | interaction: winning on reduced (0.58 vs 0.999), measured inert on full (2.245 vs 2.269) |
| warm start | [hold / OFF -- validation 4-pair verdict] | hold (speed only: solve_med 10.5 -> 2.3 s, outcomes identical) | earn-or-drop rule |
| ang_vel (omega) | 1e5 | 1e5 | NOT a free lever -- see below |
| PLAN_TIMEOUT_S | 300 (process boundary) | 300 | wall-budget; W robust (nothing observed in [35, 290] s) |
| workers | max_tasks_per_child=1 | same | aging immunity (crossover-proven) |
| envelope / bus / estimator | settled bus, unchanged | same | campaign spec |

## EXPLICITLY NOT RE-OPENABLE (findings, pinned so nobody re-litigates)

- **ang_vel stays 1e5 on both tasks.** Down-weighting it backfired (h_peak 0.267,
  final worsened): the omega cost is LOAD-BEARING FOR MOMENTUM DISCIPLINE. The tuning
  lever is angle-UP only. This is a finding, not a preference.
- **angle_N asymmetry is measured, not assumed**: terminal emphasis is the winning
  ingredient on reduced and worth nothing on full -- the sentence that justifies
  per-task configs (Section V).
- **a1e3 on full is REPORTED, not adopted**: buys ~30% more error reduction at the
  cost of most of the remaining ceiling margin (standing h 0.178 -> 0.341). The
  diminishing-returns-against-a-hard-constraint trade is presented to the reader with
  numbers; it demonstrates the Section III ceiling is actionable, not decorative.

## NOT FROZEN (out of tuning scope, unchanged from campaign)

TVLQR tracking weights (tracking measured tight at 0.011 deg -- never the problem);
window structure (500+500); all bus/sensor/estimator parameters.

## PD symmetry: the check FAILED its registered expectation -- decision needed

Registered: kp = 2.9e-4 best-or-tied on full, else report and reconsider. Measured
(kd scaled as sqrt(kp)):

| seed | kp x0.5 | kp x1 (campaign) | kp x2 |
|---|---|---|---|
| 5  | 0.947 | 0.493 | **0.382** |
| 11 | 120.4 | 105.0 | **30.2** (standing 3.1 -- late blowup, not rescued) |
| 17 | 0.394 | 0.187 | **0.112** |

kp x2 materially better on ALL THREE full-task seeds (~the tau_dist/kp offset
scaling). Context that keeps the original choice principled: 2.9e-4 was set as the
LARGEST ZERO-DIVERGENCE gain on the REDUCED task -- it is a stability-capped choice,
not an oversight; the cap binds on reduced, and full-attitude pays for it.

**DECIDED (Patrick, 2026-08-21): per-task PD gains** -- reduced keeps 2.9e-4 (its
measured stability ceiling), full gets 5.8e-4. FRAMING (better than either option as
listed): kp ~ tr(J) is a BUS rule; the data says it carries residual task dependence
-- "the scaling rule transfers across inertia but carries residual task dependence"
EXTENDS the companion paper rather than contradicting it. Seed 11 resisting the gain
fix at 30 deg corroborates authority-class, not offset-class.

**Rerun wave scope (updated):** planner-reduced + planner-full (frozen per-task
weights) + **PD-full at BOTH kp x1 and kp x2** (n=100 each, persisted per-trial --
the x1 arm exists in the campaign only as aggregates, and the registered class split
needs paired per-seed finals at both gains) + **PD-reduced at kp x1** (n=100,
persisted). The PD-reduced addition closes a SERIES GAP found by the review queue:
the money cell predates per-trial persistence, so no omega/attitude/B series exist
for it -- items 1 (demand-ratio recomputation with omega_perp + direction-appropriate
authority at instantaneous field) and 2's sigma-threshold axis are impossible without
it. ~2.5 h; PD is solver-free so reproduction of the as-run statistics is expected.

**Item-1 adjudication rule (registered pre-wave, per the review brief):** recompute
the quadrature demand ratio on the wave's PD-reduced series with (a) omega_perp
(transverse to the wheel axis) not full omega, (b) authority available in the
DIRECTION of the demand at the INSTANTANEOUS field (single-axis m_max*|B|, not the
combined-pair median ~31 uN m), (c) both peak and post-transient omega_perp
statistics. If converged trials land NEAR UNITY under the corrected normalization,
the ceiling is reported as a validated threshold; if they stay materially above
unity, it is reported as a SEPARATING COVARIATE ONLY and the residual factor is
named (candidates: standing dipole-cancellation reserve, feedforward share of the
command box). kd CONFIRMED for the draft: 8.683e-3 N m s (= 2 sqrt(KP*J_TRANS/2),
KP = 2.9e-4, J_TRANS = 0.13) -- replace the provisional 8.7e-3.

**REGISTERED before the wave (Patrick's risk flag):** if the higher gain rescues much
full-attitude non-convergence, some of what reads as architecture was tuning. The
wave reports the full-attitude failure rate at both gains, split by class:
- OFFSET-CLASS: final improves ~ 1/kp, wheel never pinned -- was tuning, not
  architecture; Section VI's full-attitude story shrinks accordingly.
- AUTHORITY-CLASS: h-pinned at either gain, or final > 30 deg unrescued at kp x2 --
  genuine frontier; stays in Section VI.
The split determines what Section VI's full-attitude claim actually is; adjudicated
from the paired cells exactly as written here.

## Validation gates (registered in TUNE_PREDICTION, adjudicated at landing)

- no easy-seed break (s3) | s78 improvement | s68 stays converged
- full cases beat their cell-2 finals materially at a1e2
- warm: earns via the 4 warm-vs-cold reduced pairs or DROPS from reduced
- sigma-conditional read: full response at 1e3 vs sigma (2-seed contrast) + reduced
  seeds' angle response vs their logged sigma
- h watch: no full-arm standing h near the ceiling at a1e2

## Results ([pending validation])

[TUNE_VALIDATE.txt + PD check tables paste here at adjudication]
