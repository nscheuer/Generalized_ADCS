# Wave D — adversarial audit of the planner-paper experiment code (2026-06-05)

Scope: twin/mismatch, disturbance, and metric-computation paths. Severity:
**HIGH** (misleads a headline claim) / **MED** / **LOW** / **CLEAN** (checked, no issue).

---

## FINDING 1 [MED-HIGH] — Twin field-mismatch is partly self-defeating (the "too good" one)
`generate_p2.8_mismatch.py`. The plan is built on the **nominal** field (`:93`
`calculate_trajectory(..., os0, ...)`), but at execution the tracker is given the **perturbed
(true) field**: `:117-118` `u = ctrl.find_u(x_hat=x, sens=true_sat.sensor_readings(os=os_k),
os_hat=os_k, ...)` with `os_k = perturbed_os(...)` (`:112`, scales/rotates B at `:104-107`).

So the magnetorquer **allocation at run time uses the measured true field** — the closed loop is
**not blind** to the field. The B-magnitude / B-rotation "mismatch" therefore only corrupts the
*open-loop plan*, which TVLQR then corrects against the correctly-sensed field. This is exactly why
the data shows B-magnitude ≈ 0° effect and 60° rotation still converges (`P2.8_mismatch_*.json`).

**Why it matters:** the honest claim is *"a plan made on a stale/nominal field stays trackable by a
field-sensing TVLQR tracker,"* NOT *"robust to field-model uncertainty."* The spacecraft measures B;
it isn't guessing it. **Recommendation:** reframe the B-magnitude/rotation rows accordingly, and lead
the twin story with the **inertia (J) mismatch**, which IS a genuine test (J is not sensed — see CLEAN-A).

## FINDING 2 [LOW] — Multi-goal "representative" figure is the best-case trial
`generate_p2.3_multigoal.py:190` — `representative_figure` selects *"the trial that converges on all
goals with smallest mean final error."* The aggregate **statistics use all 100 trials** (not a cheat),
but the single trajectory shown in `fig_multigoal_*` is the **best** one. **Recommendation:** label it
"best-case / representative," or show a median trial, so the figure isn't read as typical.

## FINDING 3 [LOW] — final-timestep metric (already addressed in B1)
The "converged" metric is final-timestep pointing error < 5° (`analyze_p2_matrix.py:76-116`). Fine for
a single-goal **hold** (matrix). It is the *same* metric that **undersold** multi-goal (anticipatory
departure → high final-in-window) — fixed by the redefined "held ≥100 s" metric (`MULTIGOAL_HELD_RESULTS.md`).
No flattering averaging window anywhere; no issue for the matrix.

---

## CLEAN-A — planner does NOT secretly see plant disturbances / true inertia / true field
- **Disturbance test** (`generate_p2_disturbance_robustness.py`): the planner's `est_sat` is built clean
  — `clean_est_sat()` sets `disturbances=[]` (`:55`); the plant adds the dipole + COM offset
  (`build_plant(True)`, `:42-50`). The planner models **no** disturbances. ✓
- **Twin J-mismatch**: `est_sat.update_J(level * true J)` (`generate_p2.8_mismatch.py:77`); plant uses
  true J via `true_sat.dynamics_for_solver`. So J **is** genuinely mismatched and **not** sensed (no
  inertia sensor) — this is the real robustness test. ✓
- **Env sweep** (`generate_p2_environment_sweep.py`): same pattern — clean `est`, disturbed plant. ✓

## CLEAN-B — no future-information / look-ahead leakage
`find_u` consumes only current `x`, current `sens`, current `os_k` — no future state or future
disturbance. The planner's legitimate foresight is over the **predictable orbit/field** (ephemeris
propagation), **not** over future plant disturbances (which are excluded from `est_sat`). The estimator
is the standard onboard filter; no oracle state is injected. ✓

## CLEAN-C — runs are genuinely paired on the same seeds
- Matrix planner-vs-PD: every generator uses `base_seed=42`
  (`generate_altro_*`, `generate_lovera_*`, `generate_lp_*` — all `base_seed=42`), so ALTRO and the PD
  baseline see the **same** random IC/goal/orbit per trial. ✓
- Disturbance test: clean and disturbed use the **same** `(g, x0, os0)` per trial
  (`:120-122`, `rand_trial(1000+i)` shared). ✓
- Env sweep: a single `trials` list reused across all environments (`:107`). ✓

## CLEAN-D — no cherry-picking / hidden filtering of failed trials
All aggregate stats iterate over **every** run (`analyze_p2_matrix.py:185-188` `errs = [... for r in
res.runs]`; multi-goal aggregates over all `trials`; disturbance/env over all N). Seeds are deterministic
(`42`, `1000+i`). No trial is dropped, thresholded-out, or reseeded-until-pass anywhere I can see. ✓

## CLEAN-E — "field fed to planner == plant field" only where legitimate
On the nominal OldPlanner path the planner is fed the **predicted** field, which equals the plant field
(BFIELD_NOTE) — that is correct foresight, not a leak (a real spacecraft propagates its own orbit). The
only place a field *mismatch* is claimed is the twin sweep, and there the issue is FINDING 1 (it's
sensed at run time), not an improper zero-mismatch.

---

## Net
The pipeline is **honest on the mechanics** — proper pairing, no filtering, no oracle/future leakage, the
planner genuinely doesn't model the plant's disturbances or true inertia. The one **framing** problem is
FINDING 1: the twin *field*-mismatch result is strong mostly because the tracker **senses** the field, so
it should be reframed (or led by the J-mismatch). FINDING 2 (best-case figure) is a labeling fix.
