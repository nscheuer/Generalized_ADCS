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

## OUT-OF-SAMPLE FOUND (Patrick's proposal, run 2026-08-21): the PD-full cell

Caveat first: the JSON lacks per-trial finals for PD-full, so the outcome variable is
h_frac_max >= 0.999 -- MOMENTUM-SATURATION events (26/100), not confirmed divergences
(the campaign table's conv5=79% implies ~21 failures; at least 5 saturation events
recovered). This is arguably the CLEANER test: saturation is the direct mechanism
outcome dump-starvation predicts; divergence is downstream.

RESULT at the frozen reduced-fit threshold (0.1035, untouched):
- catches 11/26 saturation events, 1 false alarm among the other 74 (~1%).

**WITHDRAWN (2026-08-21, Patrick's circularity catch): the "mechanism-conditional
recall" reading.** The 11-vs-15 route split ("dump-starved" vs "authority-class") was
assigned FROM THE DWELL VALUES THEMSELVES -- no independent evidence (no allocator
despin trace, no LP-saturation flags, no counterfactual was computed for these
trials). A threshold screen's catches are by construction the low-dwell tail, so
"recall = the dump-starvation share" was tautological. The route labels are removed.

WHAT STANDS (sigma-independent outcome variable only): out-of-sample on a population
the screen never saw, sensitivity 11/26 (42%) against momentum-saturation events,
precision 11/12, false-alarm rate ~1%. No mechanism decomposition is claimed.

REGISTERED path to a non-circular version (zero extra cost): the rerun wave's
PD-full cells persist full per-trial series, so an INDEPENDENT route classification
becomes computable from allocator state -- criterion registered NOW, before the wave:
dump-starved := despin commanded on >= 30% of steps AND the LP despin channel
saturated on >= 90% of those steps (the reduced-cell trace pattern: 44% / 99.7%);
authority-class := saturation without that signature. If the screen's catches then
align with the INDEPENDENTLY classified starved subset, the mechanism-conditional
sentence may return; otherwise only the recall figure survives.

Planner-full (task+controller transfer): only 2 h-pinned trials (88, 97 -- the
worker-aging fallback-contaminated pair), 1/2 flagged; 28 non-pinned >5 deg failures
are the elective-equilibrium tail, not the screen's class. Contributes little beyond
confirming planning converts divergence into graceful imprecision.
