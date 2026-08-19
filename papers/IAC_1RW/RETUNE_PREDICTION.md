# Pre-registered prediction: the TVLQR retune as a mechanism test

**Committed before Campaign A's planner half has run.** The point of this file is its git
timestamp: if the prediction below holds, the claim is "predicted, then measured," and the
history proves the order.

## Observed lean (smoke only, 420 s acquisition transient, n=1)

Plan-vs-executed deviation: median 1.08–1.11°, max ~5°. Along-B̂ energy fraction **0.451**
against an isotropy baseline of 1/3.

## Proposed mechanism

The corridor theorem, operating at the *tracking* layer rather than the goal layer. TVLQR's
plan-relative correction is itself a small along-field attitude adjustment, and the theorem
prices those: transverse deviations are corrected in seconds by direct magnetorquer authority,
while along-B̂ deviations can be corrected only by the wheel (if the cost weights let it) or by
the magnetorquers at corridor rates — slowly or effectively not at all within a tracking
timestep. Error therefore does not merely *happen* to sit along-field; it **pools** there,
because that is the direction where correction is structurally slow. Same phenomenon as the
planner paper's §IV-F observation (unmet error collects in the uncontrollable axis), one layer
down.

## The two-observable prediction

If the mechanism is right, upweighting the wheel's tracking authority along â in the TVLQR
cost must move **both** observables **together** in the retuned rerun of the two 3+1 planner
cells:

1. the along-B̂ energy fraction collapses toward the 1/3 isotropy baseline, **and**
2. the ~1° plan-deviation median collapses toward PD's ~0.16°.

| outcome | reading |
|---|---|
| both collapse together | mechanism confirmed — corridor pricing at the tracking layer; the wheel was underweighted |
| fraction collapses, magnitude does not | the weights were the wrong story; something else is soft |
| magnitude collapses, fraction does not | tracking improved for reasons unrelated to the along-field mechanism — do not cite the theorem |
| neither moves | the retune failed; the frozen-half numbers stand and the gap is not tracking |

## Preconditions before the rerun is invoked

- A's planner half (frozen weights) confirms the lean in **held-window** statistics at n=100:
  along-B̂ fraction meaningfully above 1/3 and plan-deviation median ~1°.
- Fallback fraction in those cells is low enough that they are planner measurements at all.

## If confirmed, the IV-B sentence this buys

The corridor theorem predicts that *any* feedback tracker on a magnetic bus accumulates
residual along-field — the theorem is not only about maneuvers, it is about why tracking error
has a preferred direction. A held-window along-B̂ fraction of ~0.45 against a 1/3 baseline at
n=100, from a diagnostic built for other purposes, is the theorem's fingerprint in closed-loop
data nobody designed to look for it.
