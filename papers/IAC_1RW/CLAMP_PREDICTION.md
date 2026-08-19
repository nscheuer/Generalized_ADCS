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
