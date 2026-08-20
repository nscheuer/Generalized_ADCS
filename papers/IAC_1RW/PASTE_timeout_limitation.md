# PASTE: limitations line for the replanning-timeout claim

**STATUS 2026-08-20: SUPERSEDED by Patrick's applied Section VI edit**, which corrects the
then-false "both 3+1 cells plan within budget" (the money cells wedged at ~2% of draws):
timeout affected every draw on 3+0 and ~2% on 3+1, and reported results use a hard wall
budget converting overruns into planned fallbacks -- what a flight implementation must do.
Kept for provenance; numbers below predate the money-cell wedge.

For the limitations paragraph (wherever "the fallback catches non-convergence" is cited —
likely near \ssec{planner-results}). Two-to-three sentences, drop-in:

---

The windowed replanning architecture relies on a wall-clock timeout to trigger the
feedforward fallback when a solve overruns its window. As implemented, that timeout is
enforced at the Python layer and cannot interrupt the compiled solver mid-iteration: a
pathological solve overruns silently until the solver's internal iteration caps bind, which
we observed directly when planning full-attitude maneuvers for the magnetorquer-only
configuration. The closed-loop results reported here are unaffected — both 3+1 cells plan
within budget — but a flight implementation should enforce the budget *inside* the solver
(iteration-count or per-iteration wall-clock checks) rather than at the process boundary,
since the fallback hierarchy is only as reliable as its trigger.

---

Context for you, not for the paper: this is the wedge incident of 2026-08-19 (18 workers
stuck in C++ trajOpt on the 0rw context cells, SIGALRM inert because signals fire only
between Python bytecodes). Upstream fix is queued in PR_SEASON.md item 1.
