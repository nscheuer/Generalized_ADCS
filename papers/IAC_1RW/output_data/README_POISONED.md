# POISONED_h0_* — what is invalid, and what is salvageable

`POISONED_h0_A_baseline_20260818_173649.json` + `POISONED_h0_A_rerun.log` are Campaign A's
first run, killed after the money cell. The parent computed `h0 = (1/3) * IAC_6U.h_max`
against the stale big-wheel factory (50 mN m s) while `bus_kwargs` built the settled
15 mN m s wheel, so every wheeled trial loaded **h0 = 16.7 mN m s into a 15-limit wheel**.

## Invalid for Campaign A

All `1rw_*` and `3rw_*` cells. Do not mix with the clean rerun.

## Valid as-is

The two `0rw_*` cells (n = 30 each): no wheel, so the h0 bug cannot touch them. They agree
with the clean rerun's negative controls and can corroborate them.

## Salvageable for Campaign C — an unintended anchor, with a caveat

The poisoned `1rw_reduced` cell is **n = 100 at h0 = 16.7 mN m s ≈ 2.2x the transverse
ceiling**, on the money cell's exact goal distribution, with the full covariate set
(quadrature ratio, peak omega, sigma stats) logged per trial. C's sweep tops out at
0.60 h_max = 9 mN m s, so this sits deeper into the over-ceiling regime than any planned
level — a free far-anchor for the drift-vs-h curve.

**Caveat that must travel with any use:** h0 exceeds h_max itself (111%), which no flight
wheel can store. The *dynamics* are valid (the integrator does not clamp stored momentum;
omega x h physics is exact), but the operating point is hardware-impossible. Usable to anchor
the *mechanism* curve; never quotable as a bus configuration.
