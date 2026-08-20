# Pre-registered: the planner tuning study (warm start + weights, seed 49)

Committed BEFORE the sweep produces any result (equivalence smoke + campaign cell 2 still
running at commit time). Companion to CLAMP_PREDICTION.md; git timestamp proves order.

## Measured pre-sweep, from the frozen money cell (n=99 converged, on disk)

- Standing error (held window): median 0.748 deg. **Along-B energy fraction 0.494**
  (IQR 0.339-0.621) against isotropy 1/3 -- the PLAN's own residual pools along-field,
  above even the tracking residual's 0.452. Median split: along ~ 0.43 deg,
  perp ~ 0.57 deg (per-trial medians; quadrature-consistent with 0.748).
- Jolt (replan excursion): stationary ~0.25 deg boresight / 0.42 deg 3-DOF median;
  **direction ISOTROPIC vs B** ((axis.B)^2 median 0.263, mean 0.338 vs 1/3; only 33%
  below 0.1). The bracket-maneuver (Route-A) hypothesis is REFUTED pre-sweep, as held
  loosely. Jolt is also flat vs stored momentum (corr +0.06) and vs sigma (-0.10).

## Prediction 1 -- the sweep asymptote (corridor floor in the optimizer)

If the along-field pooling is the corridor bound appearing in the plan residual, then as
`cost_main.angle` rises 1e1 -> 1e3:

- standing error falls but **asymptotes at roughly the along-field component
  (~0.3-0.5 deg)** instead of going toward zero;
- the collapse is **transverse**: the residual's along-B fraction RISES toward 1;
- returns diminish: the 1e2 -> 1e3 step buys much less than 1e1 -> 1e2.

Falsifier: standing error tracks the weight toward ~0.1 deg or below at angle=1e3 with
no asymmetry -- then the pooling was elective after all and the corridor-floor reading is
WRONG for this regime; the paper keeps "elective floor" unqualified.

## Prediction 1a -- CONTROL for the corridor reading (Patrick, 2026-08-20, pre-run)

The corridor theorem is MAGNETIC-ONLY; this is a 3+1 bus with the wheel boresight-mounted
at median sigma ~ 0.9, so the along-field direction is DIRECTLY actuated (authority
tau_w * sigma) and no corridor floor is available to the planner the way it would be on
3+0. Competing explanation for the 0.494 pooling: the momentum-cost terms (saturation /
stiction pricing of wheel excursion) make fine along-field corrections expensive --
elective, but a different weight than cost_main.angle.

Discriminating test, registered before running: correlate the along-field residual
fraction against sigma over the held window.
- CORRIDOR-LIMITED: worse at low sigma (wheel's along-field authority vanishes there;
  only the delta^(-3/2)-priced magnetic route remains). Strong NEGATIVE correlation.
- MOMENTUM-COST-LIMITED: FLAT in sigma (wheel available throughout, simply not spent).
  Then the honest sentence is "the optimizer declines to spend wheel momentum on it,"
  the corridor theorem stays in Section IV (magnetic-only), and the sweep ADDS a
  momentum-weight arm. Prediction 1's falsifier still works either way -- it then
  discriminates two ELECTIVE floors rather than elective vs physical.

### Prediction 1a ADJUDICATED (2026-08-20, same day, run after registration)

**CORRIDOR-LIMITED branch fires, strongly.** Per-trial corr(along_frac, sigma_med)
= -0.491 Pearson / -0.499 Spearman (n=99); per-sample -0.605 / -0.594 (n=9405).
Binned per-sample along-share vs sigma(t): 0.297 / 0.232 / 0.129 / **0.030** for sigma
[0,0.2) / [0.2,0.5) / [0.5,0.8) / [0.8,1). Terciles per-trial: 0.625 / 0.516 / 0.342.
(Level differs between views by theta^2 weighting; the TREND is the registered
discriminator and it is unambiguous.)

Reading: on 3+1 the corridor floor is SIGMA-GATED -- it exists exactly where the wheel
loses along-field authority. Where sigma is high the wheel scrubs along-field residual
nearly to zero (0.030); where sigma is low the residual pools along-field at the
corridor price. This is the sigma-complementarity appearing in the optimizer's residual
structure -- the FIFTH appearance of the single dial (restoration authority, corridor
escape, dump floor, dwell screen, residual anisotropy). Not momentum-cost (that branch
predicted flat); no momentum-weight arm required per the registered contingency.

REFINED sweep prediction (registered before any sweep result): the angle-weight
asymptote is SIGMA-CONDITIONAL -- high-sigma trials' standing error can fall toward the
transverse-elective level; low-sigma trials asymptote at their along-field remainder.
Seed 49 is a low-dwell (low-sigma) draw, so the single-seed sweep sees the floor case;
the validation set spans the sigma range and must show the split.

## Prediction 2 -- the jolt discriminator (warm arms)

- Jolt VANISHES under warm-hold seeding (previous plan contains no excursion) =>
  initialization character the AL iterations do not iron out (plus cost ratio jointly:
  removing it costs omega-motion priced at 1e5).
- Jolt SURVIVES warm seeding => the excursion is genuinely optimal under the weights --
  the more interesting branch; then the angle arms must shrink it if cost-ratio is the
  cause.

## Per-task tuning (DECISION, Patrick 2026-08-20; registered before any full-task sweep)

The planner gets SEPARATE weight sets for full-attitude and reduced (boresight) modes.
Justification is stronger than realism: the planner paper frames cost weights as WHAT
THE MISSION VALUES, distinct from gains that specify how; a full-attitude mode and a
boresight mode genuinely value different things, so per-mode weights are the intended
use of the interface -- what distinguishes this from per-test tuning is the freeze.

Conditions (all three binding):
1. **PD symmetry.** PD keeps ONE gain set across both tasks -- kp = 2.9e-4 follows the
   BUS-scaling rule (kp ~ ||J||, zeta = 1), a principled task-independent choice.
   To foreclose the objection, a small PD-full gain check runs: kp x {0.5, 1, 2} x
   3 full-task seeds (5, 11, 17), kd scaled as sqrt(kp) from the frozen pair.
   REGISTERED EXPECTATION: 2.9e-4 best-or-tied on full; if a factor-2 variant
   materially beats it, report and reconsider the single-gain claim.
2. **Both planner configs FROZEN before the campaign rerun**, assertion-pinned, both
   reported in the paper (pre-mission tuning, not per-test).
3. **Scope acknowledged**: second sweep (full task) + validation on both tasks.

**Full-task sweep candidate, deterministic rule:** converged pure-planner cell-2
trials, no kills, final in [3, 6] deg; pick the median-final member => **seed 35**
(final 4.23 deg, sigma_med 0.26 -- low-sigma, so the corridor component stays in view).
Same 12 CONFIGS as the reduced sweep.

**Registered prediction (price-inflation proportionality):** if the full-attitude gap
is effective-price inflation at unchanged cost ratio, raising cost_main.angle closes
it ROUGHLY PROPORTIONALLY on seed 35, as it does on seed 49. If the full case RESISTS
tuning while the reduced case responds, that is the registered transfer-failure branch
and the adjudication says so.

**ORDER DECISION (explicit, per Patrick's ask):** current chain (crossover + reduced
sweep) -> full sweep + PD-full check -> interaction tables -> per-task freeze +
validation (both tasks) -> registered n=100 rerun of BOTH money cells under the frozen
per-task configs -> **Campaign B runs AFTER, on the adopted frozen config.** Rationale:
B carries the 1/4-exponent theory test and must be generated under the same
configuration the paper reports everywhere else -- one source of truth beats two weeks
of earlier-but-inconsistent B data; the Sept 14 runway (~25 days) accommodates it.

**Campaign B dependency check (Patrick's catch, verified in code 2026-08-20):**
B runs on the equalized-torque MTQ-only bus (~32 A m^2), NOT the reference bus, so
"same config everywhere" applies only to the planner weights. Verified in
generate_B_equal_torque_planner.py: (a) slew goals are **task="full" /
Fixed_Attitude_Goal** -- the exact class that wedged the 0rw cells and killed in cell
2; (b) the file carries its OWN copy of the retired SIGALRM `_with_timeout` (line
115) -- inert against C++ grinds, so B as written can wedge exactly as the campaign
did. B-launch requirements, registered now so none is a surprise:
1. Port B's guard to `_plan_in_child` (hard process-boundary budget; overruns become
   counted fallback windows).
2. **Redesign slews to REDUCED attitude**: the 1/4 exponent is rest-to-rest rotation
   about the field axis and does not require a full-attitude goal. Construction: draw
   slew axis e PERPENDICULAR to the boresight (project the random axis onto the plane
   perp b), target = ECI_Goal(b0 rotated by Theta about e) => boresight traversal =
   Theta exactly; the |e.B| classification is unchanged. Avoids the hostile class and
   grants the optimizer the roll geometry-freedom the decomposition just quantified.
3. B's planner weights = the FROZEN REDUCED config at launch (not the stale 1e1/1e5
   block B currently carries).
4. **SMOKE GATE before the sweep**: one Theta = 0.5 rad slew on the equalized 3+0 bus,
   wall-bounded. Completes in reasonable time => campaign safe. Wedges => **flip the
   order immediately regardless of calendar** -- B then needs design work, not machine
   time, and the fallback observable is the excursion scaling measured from
   trajectories rather than converged completion times.
Reversal triggers now two: tuning pipeline slips > ~5 days, OR the B smoke wedges.

**Framing readiness (registered):** if per-task tuning closes the full-attitude gap,
Section V's crossover becomes a statement about TUNING EFFORT rather than capability
-- planner matches feedback on both tasks given per-mode weights -- and the
demonstration itself proves the price-inflation reading was elective.

## Adoption protocol (frozen before any sweep result)

1. Sweep names a winner on seed 49 (read `angle_N` terminal-emphasis arms AGAINST
   `angle` arms -- interaction, never pooled).
2. Validate on: one easy converger, frontier draws 55 and 78, **seed 68** (the planner's
   lone divergence), and one 3+3 planner trial (fully-actuated sanity).
   GATES: no easy-seed break; 68 conversion is an aspiration, not a gate.
   **AMENDED (Patrick, 2026-08-20, post-cell-2, pre-sweep-result): add one FULL-ATTITUDE
   case** -- highest-value slot now that cell 2 shows the planner-PD crossover is TASK
   CLASS (planner-full 2.23 deg pure vs PD-full 0.23), not precision. If tuned weights
   close the full gap, Section V's framing rewrites before print. TRANSFER CAVEAT,
   registered: the winner is tuned on a REDUCED cell; full attitude constrains one more
   DOF and may want a different balance. Full-case underperformance = TRANSFER FAILURE
   (flag for a full-task weight check), NOT evidence the gap is physical -- the
   adjudication must say which. Reporting rule (standing): every cell table carries
   all-trials AND pure-planner numbers with fallback fraction beside them -- the
   exclusion is defensible because both are in print and barely differ.

### Registered before running (2026-08-20): full-cell residual decomposition

One number from disk decides the corridor story's scope. Decompose the full-attitude
residual rotation vector along B (held window, theta^2-weighted, same machinery):
- pools HARDER than reduced's 0.494 => sigma story intact; full attitude demands the
  priced DOF explicitly; PD-full's 0.23 reflects horizon (an orbit vs 1000 s), and the
  causal claim stays narrow: corridor-priced WITHIN the plan horizon.
- ISOTROPIC (~1/3) => the full gap is elective equilibrium; the corridor connection was
  reduced-attitude-specific and Section V says so.

### ADJUDICATED same day (n=94 converged pure-planner): NEITHER branch clean; a third
### structure, and it names the mechanism

- Along-B fraction median **0.355** (IQR 0.223-0.503) -- at isotropy, NOT 0.494+. The
  extra full-cell error is not corridor-concentrated. BUT the sigma-gradient PERSISTS
  (corr -0.36/-0.39; terciles 0.455 -> 0.365 -> 0.235): the wheel still scrubs the
  along-field COMPONENT where sigma allows. Corridor mechanism present, not the driver.
- Boresight/roll split (the decisive number): total standing **3.023 deg = boresight
  3.023 / roll 0.030** (92/94 trials boresight-dominated). The full-attitude planner
  points the SAME boresight objective 4x worse than the reduced cell while nailing the
  newly constrained roll DOF to zero.
- READING: constraining roll removes GEOMETRY-CHOICE FREEDOM -- on the reduced task the
  optimizer picks the roll that makes MTQ transverse corrections cheapest relative to
  B; with roll pinned, the effective price of boresight correction rises and the same
  cost ratio settles at a higher equilibrium. Roll is wheel-direct (cheap) so it is
  driven to ~0; the MTQ-priced DOFs carry all the residual (marginal-cost
  equalization). ELECTIVE at bottom, but through effective-price inflation, not a new
  floor. Sweep implication unchanged (angle arms should move it); the full-task
  validation case adjudicates transfer. Diagnostics STOP here per the time-box.
3. Freeze the tuned config (assertion-pinned) and rerun BOTH 1rw planner cells n=100.
   GATES on the rerun: scheduling-rescue count >= 10/11 preserved; no new divergences
   beyond seed-68-class at base rate; fallback fraction stays ~0.
4. Paper reports tuned values plainly (pre-mission tuning, not per-test tuning), with
   the frozen-weights cells retained as the as-configured comparison.
