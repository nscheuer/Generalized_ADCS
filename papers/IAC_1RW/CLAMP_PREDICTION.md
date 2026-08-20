# Pre-registered prediction: the wheel-saturation clamp rerun

**Committed before the clamped money cell has run.** Companion to `RETUNE_PREDICTION.md`;
same rule — the git timestamp proves the order.

## What changed

`enforce_wheel_envelope`: the wheel can no longer be commanded past its momentum limit. The
unclamped integrator ran 13 of 100 money-cell trials to h = 1.84 h_max — dynamics no hardware
can produce (the same caveat as the poisoned-run README, at runtime in the clean campaign).
Sign determined empirically (h-dot = −u; the first clamp had it inverted and PUMPED the wheel
to 83 h_max, caught by the negative test).

## Prediction (Patrick, 2026-08-19, pre-data)

**Divergence persists in most of the 13, with bounded h and a different signature.** At the
diverged trials' ~0.7 °/s, even the physical cap gives ω·h_max ≈ 183 μN·m against ~31 μN·m of
transverse authority — 6× over. Saturation does not rescue those draws; it converts the
failure from unbounded ratchet to **capped gyroscopic overwhelm**.

Corroboration already in hand: the converged population's peak of 0.18 h_max is almost exactly
the rate-dependent ceiling τ⊥/ω at ~0.7 °/s (≈ 2.5 mN·m·s ≈ 0.17 h_max). **The survivors ride
the ceiling; the casualties cross it.**

| outcome | reading |
|---|---|
| most of the 13 still diverge, h ≤ h_max, overwhelm signature | prediction confirmed — frontier real, mode re-described |
| most of the 13 converge under the clamp | divergence was wholly artifact-amplified; the frontier claim retreats to the ceiling boundary C measured |
| new divergences appear in previously-converged trials | clamp interacts with desat in an unforeseen way — stop and diagnose before writing anything |

Either of the first two is publishable; the clamp decides which.

## Diagnostics attached to the rerun

- `per_trial_despin_frac` (new): does the loop ever try to despin (u·h > 0, since ḣ = −u)?
  Low + h climbing = desat channel starved by α collapse; high + h climbing = despin
  commanded but priced out of the LP box by dump-blind geometry (the D-complementarity
  closing the loop: the 3+1 desat direction needs MTQ torque along −â, which costs enormous
  dipole at high σ).
- Desat was enabled by construction in A (c_gain = 1e-3, h_target held) — the question is
  whether it was *effective*, which the sign trace answers.

## What stands regardless

The frontier's location (high-σ, near-field-axis, dump-starved draws), the near-perfect
bimodality (1 trial in 100 between 5° and 30°), the dead damping branch, and the headline
(81% within 1° vs the abstract's 73%). What the clamp decides is the failure mode's
*description* — ratchet vs overwhelm — which is exactly the sentence Section VI quotes.

Also affected and rerunning clamped: C's top two levels (0.45, 0.60 h_max), whose severity
numbers (7.9°, 24.8° RMS) carry the same artifact above the ceiling; the breakpoint's
*location* is trusted, the severity above it is not until re-measured.

## Addendum (pre-commitment, before the substitution test is read)

**What the substitution test must show to claim substitution.** The claim is the
*interaction* term, not either main effect: FF-off at h = 0.15 must show a materially lower
relative penalty versus its own h = 0 FF-off baseline than the FF-on pair shows. If only main
effects separate — feedforward better everywhere, bias worse everywhere, no interaction — the
claim retreats to "feedforward dominates bias" **without** the substitution structure, and the
Stickler & Alfriend framing softens from "obsoletes the reason for bias" to "outperforms it."

Committed before any cell of the 2×2 is read. (The test as first launched ran only the
h = 0.15 pair; the h = 0 arms were added when this criterion made clear the interaction needs
all four cells — itself an argument for writing the criterion first.)

## Interface note for PR season (the sign inversion's real lesson)

`_rw_hdot_kernel` documents the wheel-internal torque; the command is body torque; the clamp
was specified from the documented convention and inverted the physical one. The upstream fix
worth pushing is not the clamp — it is renaming or double-documenting that kernel so the
body/wheel-internal convention is unmissable at the call site. The framework thesis is that
laws port across the interface; a sign ambiguity *at* the interface is the one bug class that
thesis cannot survive. Method note for the ledger: when documentation and physics can
disagree, ask the integrator — settled empirically (3-step probe), not textually.

## Addendum 2 (pre-commitment): the reserved-desaturation rerun of the 11

The desat trace says despin is commanded and priced out of the LP box. If that is the whole
story, then an allocator that RESERVES desat authority -- pays the despin channel first and
lets pointing have the remainder -- should convert most of the 11. If the geometry starves
even a reserved channel (at high sigma the despin direction costs enormous dipole no matter
who budgets it), the reservation converts few, and the ~11% is architecture, not allocation.

| clamped outcome (11 seeds, reserved-desat allocator) | reading |
|---|---|
| >= 6 of 11 converge | the number is ALLOCATOR-priced; architecture frontier lies below 11% and VI-C's bound language stands as written |
| <= 3 converge | the LP box was not binding; frontier is architecture at ~11%, and genACS's open question answers "priority allocation does not rescue this regime" |
| 4-5 converge | split verdict; report the split |

Committed before the rerun. Either outcome answers the question genACS left open.

## Addendum 3 (pre-commitment): out-of-sample validation of the dwell screen on the planner half

The screen (flag if dwell(sigma<0.2) <= 0.1035, the LOO-stable zero-false-alarm threshold;
LOO: 9/11 caught, 1 FP, 2% residual) was fit entirely on the PD money cell. The planner half
draws IDENTICAL geometry (paired seeds), so its flagged set is known in advance:

    flagged seeds = [8, 12, 16, 23, 29, 49, 55, 78, 85]
    (computed as dwell <= 0.1035 over the money cell and frozen here BEFORE any
    planner-cell outcome is read; the diverged outliers 15 and 53 sit outside the
    screen by construction and are predicted NOT to be rescued by geometry-based
    scheduling, since their divergence is not dump-starvation)

Two independent tests when the planner half lands:
1. SCREEN VALIDATION: planner divergences, if any, fall inside the flagged set.
2. SCHEDULING RESCUE: if flagged seeds CONVERGE under the planner, dump-scheduling rescues
   exactly the draws the geometry predicts -- the D -> Section V link, measured on paired
   seeds rather than argued.

## Addendum 4 (pre-commitment, Patrick 2026-08-20): solver-hostile seeds vs the dwell screen

Committed BEFORE the hardened rerun finishes; the budget-kill log will name the wedging
seeds, and this is adjudicated exactly as written when it does.

Define W = seeds with >= 1 budget-killed window in `1rw_reduced_planner` (the cell the
flagged set F = [8,12,16,23,29,49,55,78,85] was frozen on). `1rw_full` is reported
descriptively only. Note W is a broader class than the two 14.5 h wedgers -- it includes
any solve the 300 s budget kills. Fallback events split into budget-kills vs solve-failures
vs track-fallbacks by construction (opposite implications; never one column).

| outcome | reading |
|---|---|
| W subset of F (overlap) | the geometry that starves desaturation also ill-conditions the trajectory optimization -- same underlying cause, two symptoms, and the screen predicts BOTH. A substantially stronger claim for the screen than pointing failure alone. |
| W disjoint from F | solver difficulty is an independent axis; the screen does not cover it; the paper says the two failure modes are unrelated. |
| partial overlap | report as partial. Do not reach. |

**Prior, stated now so it cannot be over-read later:** with \|W\| ~ 2 against 9 flagged of
100, chance gives P(both in F) = (9*8)/(100*99) = 0.73% and P(>= 1 in F) = 17.3%
(hypergeometric). So a full hit is strong but a single-seed overlap is weak evidence at
this n -- suggestive, not conclusive -- and is to be reported with these numbers beside it.
If \|W\| > 6, the overlap question is diluted past usefulness (chance >= 1-overlap becomes
likely and "solver-hostile" stops being a distinguished class): report W descriptively and
adjudicate nothing.

Mechanism asymmetry, registered: F was fit on PD dump-starvation dynamics; W is about
solver conditioning. An overlap is therefore a MECHANISM claim (the sigma geometry is the
proposed common cause), not a statistical artifact of refitting -- F was frozen before any
planner or wedge data existed.

### Registered discriminators (Patrick, 2026-08-20 -- still pre-data), descending weight:

1. **WHICH WINDOW wedges, not just which seed.** A window-0/1 wedge means the hard solve
   is a property of the INITIAL geometry -- overlap with F is then a direct mechanism
   claim. A late-window wedge means the solver was handed a state the trajectory had
   already drifted into -- any overlap with F is then MEDIATED by the trial having gone
   badly first, and F "predicting" it is nearly circular. Registered: a late-window wedge
   weakens the overlap claim substantially REGARDLESS of the seed-set arithmetic. (Kill
   sim-times + window indices logged per event and stored per trial: `budget_kill_t`.)
2. **W is threshold-dependent in a way F is not.** F is a property of the draw (orbit +
   goal, computable before flight); W is defined by a 300 s budget against a 500 s window
   -- at 600 s some wedgers might complete, at 150 s more would appear. Sensitivity check
   from the solve wall-time distribution (`plan_wall_s_all`, instrumented for every solve
   attempt; kills are right-censored at the budget): a clean bimodality (wedge class vs
   everything else) => W robust, threshold incidental; a smooth tail => W is an artifact
   of where the budget cuts and the overlap question is less well-posed than it looks.
   Coverage note: ~15 resumed trials predate the timing instrument and lack
   `plan_wall_s`; the histogram reads from the remainder (~185/200).
3. **Paper placement if it hits: ONE SENTENCE in the predictability paragraph, not a
   subsection** -- "the same screen also flags the draws on which the optimizer failed to
   converge within budget, on n = 2." At that n it is an observation inviting future
   work; inflating it past a sentence would undercut the discipline that earned it.

## ADJUDICATIONS 2026-08-20 -- planner money cell (1rw_reduced_planner, n=100)

Cell landed with **1200 plans, ZERO fallbacks of any kind** (no budget-kills, no
solve-failures, no track-fallbacks). Headline: 96%@5deg / 53%@1deg, median 0.99 deg,
held-p95 2.42 deg, knowledge 0.005 deg. Reads in registered order:

- **Addendum 3.1 (screen validation): FAILS AS WRITTEN, on a novel mode.** Planner
  divergences = {68} only; 68 is not in F (0/1 inside). Seed 68 is a NOVEL planner-only
  failure: converged under PD, h never pinned (end 0.32, max 0.86), sigma median 0.565,
  dwell 0.1223 (above the 0.1035 flag line, beyond the +/-0.0024 LOO band) -- not the
  dump-starvation mechanism F screens for. 12 plans, 0 fallbacks: every plan "succeeded";
  the failure is plan quality, n=1, mechanism unassigned. Report as: the screen validated
  against nothing it was built for (zero dump-starvation divergences occurred to catch)
  and missed the one novel-mode failure. No credit claimed.
- **Addendum 3.2 (scheduling rescue): 10 of 11 PD-diverged seeds CONVERGE under the
  planner** (78 improves >30 -> 8.69 deg, not converged). This includes BOTH outliers 15
  (0.17 deg) and 53 (0.94 deg), which Addendum 3 PREDICTED would NOT be rescued -- that
  sub-prediction is WRONG, and the honest reading strengthens Section V: planning rescues
  more than the geometry mechanism predicts, i.e., rescue is not screen-specific.
  Combined with ReservedDesatLP (converted 12 and 78): every one of the 11 is reachable
  by some strategy, and planning dominates reservation 10-vs-2 on paired seeds. The
  CONOPS-beats-allocation claim is now MEASURED, not argued.
- **Addendum 4: MOOT -- W is empty.** Zero budget kills in 100/100. The two 14.5 h wedges
  of the pre-hardening run did not reproduce on identical seeds/configs => the hang is a
  STOCHASTIC solver event, not a draw property. No registered branch fires. Coverage
  correction (supersedes the estimate in discriminator 2): ALL 100 cell-1 trials predate
  the timing instrument (the cell completed faster than estimated); the wall-time
  histogram comes from cell 2, fully instrumented. Operational note for VI: a
  nondeterministic hang cannot be screened out pre-flight, which if anything sharpens the
  case for the process-boundary budget.
### Addendum 4b (pre-commitment, Patrick, 2026-08-20 -- registered MID-RUN, before cell 2 is read)

Cell 2 (1rw_full_planner) has produced live budget-kills: seed 88 at window 1 (t=500)
AND window 4 (t=2000) -- the guard's first production firings, trial degrading through
counted fallbacks as designed. Registered before the cell lands:

1. **Clustering discriminator (cascade vs draw property).** Two kills in one trial's ~12
   windows is a different claim than two kills at the cell-1-suggested ~2% independent
   rate. Separator = the INTERMEDIATE windows' solve times (2 and 3 solved -- no kill
   events -- so the question is their wall time, on disk in plan_wall_s):
   - windows 2-3 at normal ~50 s => cascade WEAKENED; the draw carries the hardness
     (screenable in principle; connects the wedge class to the frontier story).
   - windows 2-3 elevated-but-under-budget => CASCADE signature: a budget overrun
     begets others through the degraded fallback state. Operationally important --
     one overrun is not one overrun.
   Chance baseline computed at read: with k kills over the cell's ~1200 windows,
   P(>= 2 in one trial | independence) by binomial; report beside the verdict.
2. **Task-class thread.** Cell 1 (reduced) closed with ZERO kills; the wedged 0rw cells
   were full-attitude; cell 2 is full-attitude and is producing kills. If cell 2's kill
   set stays confined to full attitude, Section VI extends the existing sentence from
   configuration to TASK CLASS: the tasks hardest to compute are the tasks the
   controllability analysis marks as marginal. Mechanism on hand: full attitude demands
   rank 3 on a bus that is rank-3 only where sigma > 0, and near-infeasible problems are
   where AL iteration counts blow up -- solver hostility as another appearance of the
   same underactuation, not an implementation quirk.
3. Solve-time histogram (discriminator 2 of Addendum 4) reads from cell 2's full
   instrumentation; kills are right-censored at 300 s.

- **RETUNE_PREDICTION preconditions: NOT MET -- the 2-cell retune is NOT invoked** and
  the frozen-half numbers stand, per the registration. Held-window (t >= 1000 s, n=100):
  along-B energy fraction 0.452 (smoke read 0.451; isotropy 1/3) -- the fingerprint
  CONFIRMS in direction. But deviation median is 0.011 deg, not ~1 deg: the smoke's ~1
  deg was acquisition transient. Tracking is TIGHT; the ~1 deg median final error lives
  in the PLANS (the optimizer settles ~1 deg from goal), so the corridor-at-tracking
  story is: even when tracking succeeds to 0.01 deg, the residual still pools along-field
  at 0.45 -- direction confirmed, magnitude de-fanged. The IV-B fingerprint sentence
  survives with that caveat attached.
