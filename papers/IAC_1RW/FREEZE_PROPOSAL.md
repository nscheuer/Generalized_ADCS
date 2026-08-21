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

## Results (validation landed 2026-08-21) -- GATES ADJUDICATED AS WRITTEN

sigma: reduced s3 0.37, s55 0.31, s68 0.57, s78 0.26 | full s40 0.28, s28 0.67

| arm | final (baseline) | notes |
|---|---|---|
| red s3 warm/cold | 4.56 / 5.86 (0.74) | **EASY-SEED BREAK** |
| red s55 warm/cold | 5.51 / 6.05 (1.6-2.3) | regression + h 0.55 + a BUDGET KILL (first reduced-task kill ever) |
| red s78 warm/cold | 5.76 / 5.75 (8.69) | improved, still >5 |
| red s68 warm/cold | 0.62 / 0.67 (2.90) | excellent |
| full a1e2 s40 (sigma 0.28) | 4.44 (4.11) | null -- low-sigma floor |
| full a1e3 s40 | **25.7, h=1.000, 10 kills** | CATASTROPHE: ceiling-riding realized |
| full a1e2 s28 (sigma 0.67) | 0.73 (5.83) | 8x improvement |
| full a1e3 s28 | 0.81 (5.83) | same, no extra gain |

### Adjudications

1. **REDUCED tuned candidate (a1e2/angle_N=1e4): REJECTED.** The easy-seed gate fired
   (0.74 -> 4.56), s55 regressed with a ceiling-crossing wheel (0.55) and the first
   reduced-task budget kill on record. The seed-49 win (0.58) did not transfer -- it
   was overfit to that draw. Per protocol, caught BEFORE contaminating any cell.
   PROPOSAL: reduced planner config REVERTS to the campaign baseline (angle 1e1 /
   angle_N 1e1), which is already n=100-validated by cell 1 itself (96/53/0.99, zero
   fallbacks). Consequence: the planner-REDUCED rerun is UNNECESSARY -- cell 1 stands.
2. **Warm-hold on reduced: DROPS** (earn-or-drop): at the baseline config warm is a
   tie (2.881 vs 2.907); its apparent win existed only on the rejected config.
   Warm-hold on FULL: KEPT, speed-only justification (solve_med 10.5 -> 2.3 s,
   outcomes bitwise-identical).
3. **FULL candidate a1e2-flat + warm: ADOPT (recommendation).** Three-seed evidence,
   all consistent with the sigma-conditional registration: transforms high-sigma
   draws (s28: 5.83 -> 0.73; sweep s35 standing 5.38 -> 1.71), null-no-harm on
   low-sigma (s40: 4.11 -> 4.44, within residual variation), h comfortable (0.13 /
   0.49 peak). The n=100 rerun measures the population effect properly.
4. **a1e3: REJECTED everywhere, and the s40 catastrophe is the paper's exhibit** --
   on a low-sigma draw the heavy angle weight overdrives the wheel chasing an
   along-field target the corridor prices out of reach: h pins at 1.000, every
   window's solve dies at the wall budget, 25.7 deg. "Riding the ceiling" is not a
   style critique; it is this trajectory. Strengthens Section III more than the
   duty-cycle statistic did.
5. **Sigma-conditional prediction: CONFIRMED in both task families** -- the reduced
   validation breaks are the LOW-sigma seeds (s3 0.37, s55 0.31) while the high-sigma
   seed wins (s68 0.57: 2.90 -> 0.62); the full contrast is textbook (s28 vs s40).
   Unified tuning corollary of the single dial: angle-up helps exactly where sigma
   lets the wheel serve the demand, and overdrives into the corridor wall where it
   does not.

### REVISED WAVE (pending Patrick's freeze sign-off)

| cell | config | n | purpose |
|---|---|---|---|
| planner-FULL | a1e2 flat + warm-hold, FROZEN | 100 | tuned full cell (replaces 70/15/2.32) |
| PD-full kp x1 | campaign gains | 100 | paired baseline + persistence (items 1-3) |
| PD-full kp x2 | 5.8e-4 | 100 | per-task PD + offset/authority split |
| PD-reduced kp x1 | campaign gains | 100 | series gap (items 1, 2-sigma) |

Planner-reduced: NOT rerun (config unchanged; cell 1 stands). ~10 h total,
aging-immune, one job at a time.

## APPROVED (Patrick, 2026-08-21) -- all three decisions, with framing requirement

Decision 1 framing (BINDING for the paper): both tasks were tuned per-task; ON THE
REDUCED TASK THE TUNING SELECTED THE BASELINE -- candidate fitted on seed 49, failed
multi-seed validation (easy-seed break, ceiling-crossing regression), scaling-rule
value survived. Never "reduced was untuned." Sigma-consistency sentence conditional:
if the full validation table holds the pattern (breaks low-sigma, wins high-sigma),
one sentence noting the tuning failure was predicted by the paper's organizing
quantity. WAVE IS GO: planner-full (frozen a1e2 flat + warm-hold) + PD-full x2 gains
+ PD-reduced x1, n=100 each, sequential, aging-immune.

## REGISTERED MID-WAVE (2026-08-21, before the planner-full cell completes):
## the fallback-fraction adoption line for the tuned full config

Live count at registration: 4 kill-seeds in the first ~46 dispatched (seeds 4, 11,
38, 46; windows 1/1/4/4-5) -- materially above baseline cell 2's rate, on fresh
workers. Shape note for the paper if it holds: raising the angle weight pushes toward
a BOUNDARY in whatever currency is nearest -- momentum at a1e3 (the s40 catastrophe),
COMPUTE at a1e2 (hard solves at the wall budget). Same lever, different wall.

THE LINE (set now, adjudicated at the cell table):
1. PRIMARY: if budget-kill/fallback windows exceed **10% of all windows**, the tuned
   full config is NOT ADOPTABLE -- the pointing gain is bought with PD underneath it,
   violating the cell-interpretation rule already in Section VI.
2. SECONDARY (both-ways protection): if the all-trials vs pure-planner headline
   numbers disagree by more than 5 points at conv@5, the cell is not a clean planner
   measurement regardless of the window fraction, and adoption is deferred.
3. FAILURE PATH: reconsider at angle ~3e1 (log-midpoint of 1e1 and 1e2) via a small
   bridge test on the existing validation seeds (28, 40, 35 + one easy) BEFORE any
   n=100 -- no auto-rerun.
4. Either way the cell is reported both-ways with the kill-affected trials listed,
   per the standing convention.

## AMENDMENT to the adoption-line registration (Patrick, 2026-08-21, pre-table)

1. **Gate arithmetic acknowledged**: at the observed rate (~5 kills / ~506 windows
   ~ 1%) the 10%-of-windows PRIMARY gate is non-binding; ~9% of TRIALS are
   kill-affected, so the CONCENTRATION TRIP-WIRE (both-ways disagreement > 5 pts)
   is the operative test, not the backstop. Adjudication proceeds knowing which.
2. **Comparative claim DROPPED before it enters prose.** "More hard solves than
   baseline" has no clean baseline: cell 2 predates process isolation and its kills
   were traced to worker aging. Fresh-worker baseline-weight full-task sample
   (crossover kill-seed reruns 88/94/97 + sweepfull base arms on seed 35): 7 trials,
   ~84 windows, 0 kills -- real but underpowered (P(0/7 | the a1e2 trial-level rate)
   ~ 0.45; discriminates nothing). ADOPTED: the descriptive sentence -- "the tuned
   configuration produced hard solves at a rate of [X]; the untuned cell predates
   process isolation and cannot be compared." The boundary-currency framing survives
   as a MECHANISM claim about a1e3's momentum wall (directly measured), with compute
   as a suggested parallel, not an established one.

## REGISTERED pre-table (2026-08-21): cross-controller instance-hardness overlap

Seed 11 is hard in two independent senses (PD: 30 deg at kp x2, authority-class;
planner: 3 budget kills at a1e2). The registered question: do the planner KILL-SEEDS
overlap the PD-full AUTHORITY-CLASS set beyond chance? If yes, difficulty lives in
the PROBLEM INSTANCE, not either implementation -- "hard geometries are hard for
everyone," the strongest frontier statement available.

- CRITERION: overlap of (wave planner-full kill-seed set K) with (wave PD-full
  authority-class set A, classified per the registered offset/authority rule) against
  the hypergeometric null (N=100, |K|, |A|). INFERENCE only if
  P(overlap >= observed | null) < 0.01; otherwise DESCRIPTIVE ONLY (twins
  discipline).
- POWER, stated before looking: with |K| ~ 4-6, only near-complete overlap can clear
  0.01 unless |A| is small (for |K|=5, |A|=20: E[overlap]=1.0; P(=5) ~ 1e-4 clears;
  P(>=3) ~ 0.02 does NOT). Partial overlap will almost certainly be reported without
  inference.
- CONTEXT, disclosed (old data, not the registered variable): against the kp x1
  SATURATION list (already public), the current kill-seeds {4, 11, 38, 46} overlap
  1/4 (seed 11 only), consistent with the null E ~ 1.0. Prior odds are weak; nobody
  should over-read a partial overlap later.
- CONFOUND, named: if overlapping seeds share initial-rate or goal-geometry
  covariates (|omega_0|, sigma_med, dwell), the overlap is a SHARED-CAUSE candidate,
  not instance difficulty; the covariate comparison is part of the adjudication.
