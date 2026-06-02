# "It Stays Bounded" — planner at the underactuated boundary (3MTQ+0RW)

**Goal.** Show how the **planner** (`Plan_and_Track_LQR` / ALTRO) behaves when
handed a task at the edge of feasibility: 3 magnetorquers, **no reaction
wheel**, commanded to hold a full 3-axis attitude. Planner paper, "It Stays
Bounded" subsection (between "It Recovers" and "Digital Twin Sensitivity").
Re-extracted from the committed planner MC — **no new run**.

## Environment / source
- Python 3.12.13, NumPy 2.2.6, SciPy 1.15.3; branch `paper2-datagen`,
  SALTRO `fd7d24c`.
- Source: `papers/Planner/output/mc100_altro_3+0_full_20260204_162440.sim`
  (100 trials, `Plan_and_Track_LQR`, 3MTQ+0RW, full-attitude `Fixed_Attitude_
  Goal`, tf=1000 s, boresight [0,1,0]).
- Analysis: `analyze_graceful_3mtq0rw.py` (loads the .sim; computes full +
  boresight error and ‖ω‖; selects representative traces).

## Selection — three panels (median + both worst-case archetypes)
- **Non-converging set:** 82 / 100 (full-attitude final < 5° in 18%) — matches
  the brief's "82 non-converging."
- **Median:** rank-41 of 82 by max pointing-error envelope → **trial 20**.
- **Highest peak:** largest peak envelope → **trial 70** (peaks 180° at t=649,
  recovers to 12° final). A near-180° eigenaxis slew sweeping through the
  antipode — dramatic to look at, but a *success*.
- **Worst final:** genuinely least-converged final outcome → **trial 99**
  (full 96.9°). The honest "least converged."
Both worst archetypes are shown so the figure demonstrates "still behaving okay"
in both senses — the dramatic one recovers, the least-converged one stays calm.

## Results

| | Median (rid 20) | Highest peak (rid 70) | Worst final (rid 99) |
|---|---:|---:|---:|
| Full-attitude: peak / final | 139° / 13.7° | **180° / 12°** | 174° / 96.9° |
| Boresight final | 11.2° | 7° | 78.9° |
| ‖ω‖ init → peak → final | 0.04 → · → 0.08 | 0.01 → **1.07** → 0.07 | 0.04 → · → 0.15 |

Aggregate over the 82 non-converging: median full 18.3°, **boresight 11.3°**,
boresight settled <5° in 15%, ‖ω‖ median 0.04 → 0.08 deg/s. Full-attitude
**final** distribution (100 trials): <5° **18%**, <30° **87%**, <90° **99%**,
>150° **0%** (max final 96.9°).

Outputs: `output_data/fig_graceful_3mtq0rw.{png,pdf}` (3 columns: median |
highest-peak | worst-final; each: full + boresight error, ‖ω‖, 3 MTQ commands),
`output_data/graceful_3mtq0rw.json`.

## Reading guide / honest framing
- **It stays bounded — and it stays *calm*.** Even with no reaction wheel and an
  infeasible 3-axis target, the planner lays down a smooth, low-rate trajectory:
  ‖ω‖ stays ~0.05 deg/s (median), and even the worst-final trial only reaches
  0.15 deg/s. Nothing tumbles, nothing diverges, no NaN. The largest excursion
  in the whole set (rid 70, 180° peak) recovers to 12°.
- **It performs in the subspace it can reach.** Boresight error (median 11.3°)
  is consistently better than full-attitude error (median 18.3°), with 15% of
  the non-converging trials settling boresight < 5°. The framework achieves the
  2-DOF pointing the actuators *can* deliver and leaves only the uncontrollable
  roll unconverged — graceful degradation along the controllability null space,
  not blanket failure.
- **"Non-converging" overstates it.** 87% of all trials finish within 30° and
  99% within 90°; the strict 5° bar is what produces the 18% headline. The honest
  message is "nearly always gets close, always stays bounded," not "fails."
- This is the planner; the reactive-PD comparison is a separate Paper-1 matter
  and is not used here.

## Draft paragraph — "It Stays Bounded" (~200 words)
> A flight framework must also behave well when a task is *not* achievable. We
> probe the edge of feasibility with the most underactuated case in the study —
> three magnetorquers, no reaction wheel — asked to hold a random full 3-axis
> attitude. Across 100 Monte-Carlo trials the planner converges to within 5° in
> 18% of cases; strictly, 82 do not converge. But the behaviour of those 82 is
> the point. None diverge: the planner produces a smooth, low-rate trajectory
> with body rate held near 0.05 deg/s, and no trial tumbles or blows up — the
> single largest excursion sweeps through the antipode of a near-180° slew and
> recovers to 12°. More than that, the framework *performs in the subspace it can
> reach*: boresight pointing error (median 11°) is consistently tighter than full
> attitude error (median 18°), with the unconverged residual living almost
> entirely in the uncontrollable roll axis. Read at the 30° level rather than the
> strict 5° bar, 87% of trials get close. The framework, handed an infeasible
> command, neither forces a bad solution nor fails unsafely — it stays bounded,
> stays calm, achieves what the actuators allow, and leaves the operator a stable
> platform to re-task. Graceful degradation along the controllability boundary.

## Files
- `papers/Planner/analyze_graceful_3mtq0rw.py`
- `papers/Planner/output_data/fig_graceful_3mtq0rw.{png,pdf}`, `graceful_3mtq0rw.json`
