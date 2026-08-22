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

## Bridge test STAGED (pre-table, per Patrick): angle ~3e1 retreat path

Kill-seed count is tracking well past the trip-wire's likely firing point, so the
bridge is staged now: --bridge mode, angle=3e1 flat (log-midpoint) + warm-hold, full
task, seeds {28 (high-sigma), 40 (low-sigma held-out), 35 (tuning seed), + the
lowest-final no-kill cell-2 converger picked in-mode (deterministic easy case)}.

REGISTERED bridge gates (before any bridge data): PASS iff
(a) s28 final <= half its cell-2 baseline (2.9 vs 5.83) -- the high-sigma win must
    survive the retreat;
(b) s40 no worse than its baseline 4.11 (low-sigma null tolerated, harm not);
(c) easy seed stays converged;
(d) kills across the four trials <= 1.
PASS => freeze full config at 3e1 and rerun ONLY the planner-full cell (others
stand). FAIL => full planner reverts to baseline weights (symmetric with reduced:
"tuning selected the baseline" on both tasks), and Section V reports the boundary-
currency finding: the angle lever is walled on BOTH sides for this solver.

## FRAMING CORRECTED (Patrick, 2026-08-21): both walls are ABOVE baseline; compute binds FIRST

Not "walled on both sides": the momentum wall appears at a1e3 (standing h 0.34, 4.4%
of orbit above ceiling) and the compute wall at a1e2 (~14% of trials with kills) --
both above baseline, with COMPUTE ARRIVING FIRST, at a weight the momentum ceiling
still tolerates comfortably (h_peak 0.278 vs 0.42). The engineering sentence: a
practitioner tuning this planner hits solver reliability before the physical
ceiling; the ceiling is the SECOND constraint encountered. If the bridge at 3e1
clears the compute gate while keeping half the pointing win, the compute boundary is
located to within a factor of ~3 (between 3e1 and 1e2) -- a real number for Section V.

## DECIDED (Patrick's lean, executed): clean baseline planner-full cell added -- UNCONDITIONALLY

Cell E joins the wave (runs on the post-wave resume pass): baseline weights, stock
controller, cold -- exactly cell 2's configuration on the immunized harness. Needed
in EVERY branch: bridge-pass -> it is the as-configured baseline comparison; bridge-
fail -> it is the adopted config's own cell. Cell 2 retires to "pre-isolation era,
superseded" with the caveat stated once. The awkward sentence ("our planner-full
numbers came from a configuration later found systematically contaminated") is never
written.

## ADOPTION-LINE ADJUDICATED (2026-08-21, cell table): PASSES BOTH GATES -- ADOPTED

planner-full frozen a1e2+warm, n=100: ALL 94.0/23.0/1.42; PURE-PLANNER (68 trials)
92.6/19.1/1.47. PRIMARY gate: fallback windows 6.5% < 10% PASS. TRIP-WIRE: conv5 gap
1.4 pts < 5 PASS (conv1 gap 3.9; medians 1.42 vs 1.47).

The surprise the gates were built to measure: kill-affected trials number 32 (78
kills total -- far beyond the monitor-visible subset), yet their finals are FINE
(median ~1.2 deg, one >5 at 7.8): a killed window degrades to fallback for 500 s,
the next plan recovers. Window-level contamination 6.5%, trial-level 32%,
outcome-level ~nil. The paper prints both-ways + the kill accounting; the
"hard solves at a rate of X" descriptive sentence uses 78/1200 windows and 32/100
trials, no baseline comparison until Cell E lands.

vs the RETIRED contaminated baseline (context only): 72.2/15.5/2.23 -> 94/23/1.42.
The printed comparison awaits Cell E. BRIDGE: stays staged, UNUSED -- its trigger
did not fire. Wave continues: E (running) -> PD x3.

## CELL E LANDED (2026-08-21): cell 2 vindicated; the comparative kill sentence RESTORED

Clean baseline planner-full (seed-paired, immunized harness): **72.0/19.0/2.24,
4 kills / 1200 windows (0.33%)**.

1. **Cell 2 was fine all along**: its pure-planner subset read 72.2/15.5/2.23 -- the
   clean rerun reproduces it to within tenths (conv5 72.0 vs 72.2, median 2.24 vs
   2.23). Aging cost ~nothing in surviving trials; the retirement caveat shrinks to
   one clause, evidence-backed. Per-seed paired deltas to the ledger at wave end.
2. **The comparative hard-solve sentence is SUPPORTABLE again**, now with same-harness
   provenance: baseline 4 kills vs a1e2's 78 per 1200 windows -- a ~20x increase from
   a 10x weight change. The compute-wall claim gets its number.
3. **The adopted config's printed comparison** (same harness, same seeds): 94.0/23.0/
   1.42 vs 72.0/19.0/2.24 -- +22 pts conv@5, median 1.6x better. Per-task tuning
   delivered on full attitude, against a clean baseline.

## WAVE READS ADJUDICATED (2026-08-22) -- six registered items

A. **Offset/authority: 31 authority, 0 offset, 0 residual.** kp x2 rescued nothing
   net (failures 22 -> 27: it BREAKS 5 marginal draws while halving convergers'
   median). NOTHING that reads as architecture was tuning -- the strongest available
   resolution of the registered risk. Section VI's full-attitude story is
   architecture, with the gain trade stated.
   DECISION FLAG for Patrick: per-task PD-full at 5.8e-4 now has a measured trade
   (73/71/0.13 vs 78/72/0.26) -- not dominant. Which arm prints (or both) is a
   claim-affecting choice.
B. **Route classification: STARVED = EMPTY -- the mechanism-conditional withdrawal
   STANDS.** The registered allocator criterion (despin >= 30%, bind|despin >= 90%)
   fires on nobody: rerun binding fractions run 0.60-0.90, far from the original
   trace's 99.7%, so the criterion's operationalization (any-MTQ-at-box) evidently
   differs from the original diagnostic's saturation measure. Per discipline: no
   post-hoc criterion; the recall claim stays withdrawn for the paper. Rerun diverged
   set: 12 (original 11 + seed 62); screen catches 8/12 @ 0 FP on fresh labels.
C. **Demand ratio: VALIDATED AS A THRESHOLD under the corrected normalization.**
   Post-transient: diverged ride the boundary (median 0.89 ~ unity), converged sit
   far inside (0.00, IQR 0.00-0.01). Peaks are slew-dominated for everyone (5.1 vs
   41.8) -- the old "both populations above unity" was full-omega + median-authority
   artifact. The ceiling is a threshold in the post-transient statistic; peak is slew
   context. The registered residual-factor question dissolves.
D. **Sigma-threshold axis: the pre-outcome 0.2 cutoff is locally optimal** -- at
   sigma<0.2/cut 0.1035: 8/12 @ 0 FP; sigma<0.15 collapses (21 FP), sigma<0.25
   loses catches. Knife-edge structure confirmed on fresh labels.
E. **Hardness overlap: NULL** -- overlap 10 vs E=9.9, P=0.57. Planner-hard and
   PD-hard are INDEPENDENT populations; "hard geometries are hard for everyone" is
   refuted, one sentence, descriptive per registration.
F. **Aging deltas (ledger): cell-level reproduction excellent (median |delta| 0.073
   deg), per-seed tail bistable (p90 18.5, max 165)** -- ~10% of draws flip outcome
   between eras while cell statistics match. Cell stats robust; per-seed caveat
   already in place.
