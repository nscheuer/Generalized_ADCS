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
