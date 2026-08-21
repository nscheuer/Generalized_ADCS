# A3: screen provenance, runner-ups, out-of-sample status (2026-08-21)

## Provenance (git-dated)
- 92b4b56b 08-18 12:19 "log the geometry covariates A's divergence correlation needs":
  sigma_dwell_below_0p2 -- WITH its 0.2 cutoff -- enters the per-trial schema, BEFORE
  the clamped money cell landed (08-19) and before any clamped divergence existed.
  Nominated from the III-D / Campaign-D dump-window mechanism (time in dump-capable
  geometry). The registered mechanism fork (damping vs frontier geometry) predates the
  read in CLAMP_PREDICTION.
- f4ec714b 08-19: screen built, threshold 0.1035 fit on the 11 outcomes, LOO'd.
- PRECISE STATEMENT: the covariate family (sigma-geometry) and the 0.2 cutoff were
  mechanism-specified and instrumented pre-outcome; the choice of dwell AMONG the
  three logged sigma summaries (dwell / sigma_med / sigma_min), and the threshold,
  were made after the outcomes were read (threshold LOO-covered; family selection not).

## Runner-ups (all 22 logged candidates, catches at <= 1 FP, same rule, LOO on events)
Perfect/near separators -- ALL outcome-contaminated (in-flight measurements that are
failure signatures, not screens): tracker_avail 11/10 (lock lost while tumbling),
alpha_median 11/10, alpha_frac 11/10, h_frac_max 11/10 (divergence IS h-pinning),
peak_omega 11/10, damping_ratio 11/10, quadrature_ratio 11/10, est_att_err 10/9.
PRE-FLIGHT-COMPUTABLE family (orbit+goal geometry only): **dwell 9/8**, sigma_min 6/5,
sigma_median 2/1, eclipse_frac 0/0.
=> Among candidates usable as a pre-flight screen, the mechanism-named variable
DOMINATES its family; the 11/11 separators are echoes of the failure, not predictors.

## Out-of-sample status: VACUOUS, cannot be leaned on
The pre-registered planner-half test adjudicated with zero dump-starvation divergences
available (10/11 rescued; the single novel-mode miss was later reclassified as a
worker-aging artifact). No usable out-of-sample validation exists or is scheduled.

## Replication note
Re-running the LOO here gives dwell 9/11 in-sample @ <=1 FP and 8/11 LOO vs the
reported "LOO 9/11 @ 1 FP" -- fold-rule details matter at this n; the paper must
state the exact procedure and quote figures from one stated rule.
