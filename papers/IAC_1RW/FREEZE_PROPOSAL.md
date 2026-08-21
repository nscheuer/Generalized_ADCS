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
PD gains (single bus-rule set, task-independent, PD-full check = [pending]);
window structure (500+500); all bus/sensor/estimator parameters.

## Validation gates (registered in TUNE_PREDICTION, adjudicated at landing)

- no easy-seed break (s3) | s78 improvement | s68 stays converged
- full cases beat their cell-2 finals materially at a1e2
- warm: earns via the 4 warm-vs-cold reduced pairs or DROPS from reduced
- sigma-conditional read: full response at 1e3 vs sigma (2-seed contrast) + reduced
  seeds' angle response vs their logged sigma
- h watch: no full-arm standing h near the ceiling at a1e2

## Results ([pending validation])

[TUNE_VALIDATE.txt + PD check tables paste here at adjudication]
