# Pre-registered: the VI-F "approaches three-wheel performance" margin

Criterion FIRST (registered before adjudication): "approaches" is printable iff the
3+1 reduced cell's median final error is within 3x of the 3+3 cell's AND the conv@5
gap is <= 15 points with the n=30 context-cell binomial interval quoted. Otherwise
the sentence becomes "trails three-wheel performance by [X] points / [Y]x".

## Adjudication -- with a correction on the record

First adjudication used a from-memory 3+3 median of 0.16 deg and passed; the VERIFIED
number (A_baseline_20260818_202627, settled-bus clean grid, the last run containing
the 3rw context cells) is **0.076 deg** -- the memory figure was wrong and the pass
was retracted within the hour.

Verified table: 3+1 reduced 89.0 / 82.0 / 0.25 (n=100, clamp-era) vs 3+3 reduced
100.0 / 96.7 / **0.076** (n=30, pre-clamp era; comparable -- the 3-wheel cells peaked
at 0.13 h_max, far from any envelope, so the clamp is inert for them).

- median ratio 0.25 / 0.076 = **3.3x  -> FAILS the within-3x criterion**
- conv@5 gap 11.0 pts (n=30 binomial sigma ~3.3 at p~0.97) -> passes

**VERDICT, as registered: "approaches" is NOT printable.** VI-F's sentence is the
quantified form: "the single-wheel bus trails the three-wheel reference by 11 points
in 5-deg convergence (89 vs 100%) and a factor of 3.3 in median pointing error
(0.25 vs 0.076 deg, n=30 context cell)." The margin is defined, quoted, and closed
against "how close is approaching?" -- by conceding the adjective and printing the
numbers.
