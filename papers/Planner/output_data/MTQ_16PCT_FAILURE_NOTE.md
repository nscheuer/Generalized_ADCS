# 3+0 (MTQ-only) reduced: the 16% non-convergence, diagnosed

Source: `mc100_altro_3+0_reduced_20260204_185357.sim` (100 trials, no new run). 84 converged, 16 failed.

## Naive hypothesis REFUTED
- slew-phase blocking |n̂·B̂|: converged 0.48 vs failed 0.48 (identical)
- initial error: converged 56° vs failed 54° (one failed trial started 12° away)
So failures are NOT 'goal near the uncontrollable axis' or 'harder slews'.

## Real mechanism: two populations
1. **Horizon truncation (5):** end-slope < -5 deg/1000s (still slewing fast); large finals (up to 27°). Longer horizon would converge.
2. **Terminal MTQ-hold limit (11):** flat/positive end-slope; finals 5–89°; terminal blocking 0.72 vs 0.53 (converged). The residual correction torque sits near the instantaneous field, which a magnetorquer cannot produce, so it drifts/limit-cycles rather than holding.

Terminal blocking overall: converged 0.53 vs failed 0.67.
