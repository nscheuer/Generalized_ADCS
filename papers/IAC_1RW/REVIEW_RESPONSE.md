# Pre-submission review responses (2026-09-07)

## Item 1 — LP alpha collapse + QP MTQ-only cell

**LP check (one sentence):** the LP's equality constraint is
`A_total u = T_avail tau_hat` with NO projection (`allocate_max_torque_in_direction`,
mtq_w_rw_LP.py:798-905); MTQ torque lies perp-B by construction, so any request with a
nonzero along-field component admits only T_avail = 0 -> **alpha = 0 exactly. The
reviewer is right; the QP cell is mandatory.**

QP arm: 3MTQ+0RW reduced PD, n=30, SAME seeds as the context cell, settled bus,
identical dipole feedforward via MRO mixin (`FeedforwardQP(FeedforwardLP, MTQ_w_RW_QP)`
-- FeedforwardLP is super()-cooperative). Result (paired seeds, one orbit): **conv@5 0.0%, conv@1 0.0%, median 54.3 deg,
div>30 90%** (best trial 6.0 deg) vs the LP context cell's 0/0/74.0. The QP makes
progress the LP structurally cannot (median 54 vs 74) and still converges on NOTHING
within the orbit. **Headline sentence: with either allocator, MTQ-only converges on
0% within one orbit** -- a timescale result, not an allocator artifact, consistent
with Campaign B's torque-invariant floor.

## Item 2 — provenance of the 57% / 32% along-axis shares

**Verdict: (b) COMPUTED** — `accum_along_wheel = |secular_rate . a_hat|` from the
disturbance-model integral (generate_F_altitude.py:180); F is model-side and has no
closed-loop h(t). Reproduced exactly: nadir 0.994 mN m s/orbit (57% of |total| 1.755),
inertial 1.089 (32% of 3.408).

**Reviewer's torque-balance quantity** int (B_hat . tau)/sigma dt, sigma floored at
0.05 (trace min 0.004 nadir / 0.000 inertial): secular 0.139 (nadir) / 0.185
(inertial) mN m s/orbit — 5-7x SMALLER than the projection. Within-orbit peak of that
integral ~0.28-0.29.

**Measured closed loop (the decisive check):** wave PD cells, converged trials, one
orbit: net |dh| median **1.103 mN m s/orbit** (both reduced and full) — within 1.3% of
the inertial a_hat-projection (1.089). The projection empirically predicts what the
wheel actually absorbs under the campaign's allocation; the idealized balance does not,
because the closed loop does not run the MTQ-takes-everything-perp-B split (the LP
gives the wheel its axis share directly). Caveat: the measured check covers
inertial-type goals only; no closed-loop nadir cell exists, so the 57% nadir figure
stands as model-side with the derivation text adjusted (present D's projection as
validated-by-measurement, discuss the balance form as the idealized alternative).

**Peak vs net (reviewer's excursion point):** peak |h| median 2.75 / 2.68 mN m s
(18.3% / 17.9% h_max) = **2.1x / 2.6x the per-orbit net** — same-orbit saturation
margin should budget the excursion (~2.5x net); multi-orbit saturation is still paced
by the net.

## Item 3 — saturation enforcement (one sentence + audit)

The wheel envelope (`enforce_wheel_envelope`: u clipped to [(h-h_max)/dt,
(h+h_max)/dt], hdot = -u established empirically) is applied in the HARNESS with true
h (`_iac_sim.py:669`) and controller-side in FeedforwardLP, since commit `11bdb1d9`.
Coverage: all 1rw grid cells (PD + planner, reduced + full), C's clamped rerun, all
wave/validation/tuning cells. NOT applied in Campaign B's runner — measured
IMMATERIAL: 3+1 slews start at h=0 and peak at 0.018-0.037 h_max (probed at
Theta = 0.5 and 2.0; h tracking now added to run_slew). D and F are model-side (no
closed loop; N/A). Pre-clamp cells, named: the 8-18 context cells — 0rw (no wheel;
vacuous) and 3rw (peaked 0.13 h_max, envelope never approached).

## Item 4 — hold metrics + intervals (final 500 s; Wilson 95% on fractions; 5000-resample bootstrap on medians)

| cell | n | conv5 [CI] | conv1 [CI] | median [boot CI] | hold<1 (med frac of final 500 s) | hold<5 | p90(final 500 s) |
|---|---|---|---|---|---|---|---|
| PD 3+1 reduced | 100 | 88 [80,93] | 82 [73,88] | 0.23 [0.20,0.30] | 1.00 | 1.00 | 0.31 |
| PD 3+1 full | 100 | 78 [69,85] | 72 [63,80] | 0.26 [0.19,0.38] | 1.00 | 1.00 | 0.46 |
| planner 3+1 reduced | 100 | 96 [90,98] | 53 [43,62] | 0.99 [0.89,1.07] | 0.72 | 1.00 | 1.36 |
| planner 3+1 full (tuned) | 100 | 94 [88,97] | 23 [16,32] | 1.42 [1.31,1.58] | 0.41 | 1.00 | 1.88 |

Converged PD trials hold sub-degree for the ENTIRE final 500 s (median fraction 1.00,
p90 0.31/0.46 deg) — "sustained sub-degree" is demonstrable, not just terminal. The
planner cells hold <5 deg fully; the <1 deg hold fraction (0.72/0.41) is the ~1 deg
plan equilibrium in hold form.

Paired per-seed outcomes (same seeds by construction):
- reduced @5deg: both 85, PD-only 3, planner-only 11, neither 1
- reduced @1deg: both 44, PD-only 38, planner-only 9, neither 9
- full @5deg: both 73, PD-only 5, planner-only 21, neither 1
- full @1deg: both 16, PD-only 56, planner-only 7, neither 21

Context cells (0rw/3rw): no per-trial series persisted (8-18 JSON aggregates only) —
hold metrics unavailable for them; their cell-level held-p95 aggregates exist in the
JSON.

## Item 5 — figures

`fig_seed8_threeway.{pdf,png}` (PD divergence / desaturation-first exchange / planner
rescue; err log, |h|/h_max with h_max line, sigma with 0.2 threshold, LP alpha) and
`fig_pd_hist.{pdf,png}` (final-error histogram, log x, 5 and 30 deg lines). Seed 8 finals:
PD 85.4 deg (wheel pinned at h_max from t~0.2 hr), desaturation-first 124.9 deg
(h held near ZERO all orbit -- momentum objective met, attitude lost: the exchange),
planner 0.5 deg (h low AND converged). All three series persisted (reserved-desat
rerun 2026-09-07).

## Not run

Planner-to-PD handoff test — presented as the next test, per the brief.
