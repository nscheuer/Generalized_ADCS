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
3. Freeze the tuned config (assertion-pinned) and rerun BOTH 1rw planner cells n=100.
   GATES on the rerun: scheduling-rescue count >= 10/11 preserved; no new divergences
   beyond seed-68-class at base rate; fallback fraction stays ~0.
4. Paper reports tuned values plainly (pre-mission tuning, not per-test tuning), with
   the frozen-weights cells retained as the as-configured comparison.
