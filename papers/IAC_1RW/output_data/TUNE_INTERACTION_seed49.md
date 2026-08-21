# Seed-49 (reduced, low-sigma floor case) sweep -- interaction reading

anchor: base_cold final 2.907 vs money-cell 2.80 / fresh-spread 2.76-2.93 -- decomposed
path drift concern CLOSED.

## Jolt (Prediction 2): COST-STRUCTURE, not init character
- warm seeding does NOT remove it (0.350 -> 0.325): the optimizer re-introduces the
  excursion from a no-excursion seed => genuinely optimal under 1e1/1e5.
- angle x10 KILLS it (a1e2: -0.009): the more-interesting registered branch, closed.

## Asymptote (Prediction 1, sigma-conditional): CONFIRMS on the floor case
- standing: base 0.898 -> a1e2 1.006 -> a1e3 1.249. The held floor DOES NOT MOVE with
  angle weight on this low-sigma draw (falsifier did not fire; asymmetric response).
- final improves 2.91 -> 1.75 by jolt removal, not floor movement.
- diminishing returns as registered: 1e1->1e2 buys 1.16 deg; 1e2->1e3 buys nothing.
- cost: solve_med 7.5 -> 33-43 s at a1e3 (planner paper's 8x warning reproduces).

## Interaction (angle_N read AGAINST angle, never pooled)
- terminal emphasis ALONE FAILS: termN1e3 (1e1/1e3) final 3.22 > base 2.91.
- combined it is the WINNER: a1e2N1e4_warmhold final 0.582, err@5400 0.548,
  jolt 0.227, 0 kills, solve_med 23 s, h_peak 0.180.
  vs a1e2_warmhold (flat 1e2): 0.999 -- angle_N x100 nearly halves it again.
- warm-hold synergizes with higher angle (a1e2: cold 1.75 vs warm 1.00) though not at
  base; warm does NOT cut solve time (expectation refuted).
- ang_vel down-weighting HURTS (a1e2_av1e4: 2.56 final, h_peak 0.267 -- omega cost is
  load-bearing for momentum discipline). Ratio lever works via angle UP only.

## Candidate frozen REDUCED config (pending registered validation + gates)
angle=1e2, angle_N=1e4, ang_vel=1e5, warm-hold. 5x better final on the floor draw,
jolt eliminated, budget margins comfortable. Gap noted: a1e2N1e4_cold was not run;
warm-hold is part of the candidate as-swept.

## Pre-freeze protections (Patrick, 2026-08-21)

- **Warm-hold must earn its place or be dropped.** It has failed both original
  rationales (no solve-time cut; no jolt removal). Its one remaining measurable
  contribution is solution quality at high angle (a1e2 flat: cold 1.75 vs warm 1.00).
  PENDING: a1e2N1e4_cold x2 (running) vs the swept warmhold 0.582 -- if cold matches,
  the frozen config drops warm; every component earns its place.
- **Omega-lever design rule (paper keeper):** the omega cost is load-bearing for
  momentum discipline (down-weighting it raised h_peak to 0.267 and worsened final);
  the tuning lever is angle-UP only. One line, connects to the momentum channel.
- **Nondeterminism phrasing lock:** "deterministic on fresh workers, with residual
  variation on some draws" -- 3/68 bitwise vs 49/55 spread 0.17-0.78 is not noise and
  not stochasticity; do not call it either way in print.
- **Section V simplifications:** seed-68 divergence + novel-mode discussion OUT
  (harness artifact); per-seed caveat retired.
- **Section VI tuning sentence w/ mechanism:** terminal emphasis only helps once the
  running attitude cost makes the pull meaningful (angle_N alone: 3.22 WORSE than
  2.91 base; combined with angle=1e2: 0.582).
