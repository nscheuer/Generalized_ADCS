# Review notes — content & framing (2026-06-04)

Substantive suggestions from working the code + results, beyond typos/placeholders (those
are in `TEX_CHANGES_SUGGESTED.md`). Nothing applied to the `.tex`. Confidence tagged:
**[verified]** = checked against code/results this work; **[judgment]** = editorial.

---

## Paper 2 (Planner)

### 1. The disturbance claim is now BACKED by an explicit test (was: near-zero MC disturbance)  **[verified — test added]**
**Original finding:** the *nominal* MC plant has essentially no disturbance torque — GG peaks
~$4.3\times10^{-8}$ N·m, and drag + SRP are exactly zero (no cp–cg offset; verified `tau = -0.0`). So the
abstract clause *"under disturbances excluded from the planner's model"* was not actually exercised.

**Resolution (preferred over softening): a dedicated test now substantiates it.**
`fig_disturbance_robustness` + `DISTURBANCE_ROBUSTNESS_RESULTS.md`
(`generate_p2_disturbance_robustness.py`): the 3+1 reduced testbed, paired by seed (N=20, $t_f=1000$ s),
clean plant vs. a plant carrying **real** disturbances the planner does **not** model — a residual magnetic
dipole ($|m|=0.028$ A·m² → **peak $1.1\times10^{-6}$ N·m**, the dominant term), aero drag and SRP made
non-zero by a 2 cm cp–cg COM offset (~$5\times10^{-9}$ N·m), and GG. Result:

> convergence **100% → 100%**; mean final boresight error **0.11° → 0.22°**.

So the planner+TVLQR reject ~1 µN·m of unmodeled disturbance with no loss of convergence and sub-degree
pointing. The abstract clause is now literally true. (The P2.8 mismatch sweep and the P2.6 leak remain the
*model-error* robustness story; this closes the *disturbance* half.)

**Draft abstract clause** (now defensible as written, or sharpen to):
> "...with random orientation, goals, rate, and orbital parameters, under gravity-gradient, aerodynamic, SRP,
> and a residual-magnetic-dipole disturbance ($\sim\!1\,\mu$N·m, dominant) absent from the planner's model;
> convergence is unchanged and final pointing stays sub-degree."

### 2. State the disturbance-observability limit (it's a real, publishable point)  **[verified, from P2.6 work]**
The compensation block and the extended TVLQR (Eq.~\eqref{eq:tvlqr_extended}) rely on *estimated* disturbance
torques from the augmented-state estimator. But a **constant body-fixed torque that the wheels absorb is not
observable from attitude alone**: it is soaked into a constant wheel-momentum offset and produces no attitude
signature. The observable quantity is the momentum *rate*, and separating a constant leak from a bias requires
geometry change (eclipse boundaries, star-tracker updates). Acknowledging this both pre-empts the obvious
reviewer question and is a genuine estimation-side contribution.

**Draft sentence** (end of \S\ref{subsec:scheduling} or the compensation disturbance-feedforward paragraph):
> "One caveat bounds what feedforward can correct: a constant body-fixed disturbance absorbed into a constant
> wheel-momentum offset is unobservable from attitude alone---it produces no attitude signature, and only the
> momentum \emph{rate} (revealed as orbital geometry changes) distinguishes it from an actuator bias. Slowly
> varying or attitude-coupled disturbances are recovered; a perfectly constant wheel-absorbed leak is not,
> and is instead handled by the wheels until momentum management acts."

### 3. Use the momentum-dump spike as the concrete cost-tuning example  **[verified, from SALTRO tuning]**
\S\ref{subsec:cost} notes abstractly that "terms can work against each other." A specific, memorable instance:
an over-stiff reaction-wheel momentum weight produced a *converged* plan that dumped momentum in a single
body-torquing lump (a transient pointing spike of several degrees); reducing the weight let the wheels hold
the offset and held sub-degree. Worth one concrete sentence — it also reinforces that **a converged solve can
still yield a poor closed loop, so the failure is in the plan/cost, not the tracker** (which justifies the
simple TVLQR tracker in \S\ref{subsec:following}).

**Draft sentence** (\S\ref{subsec:cost}, after "must be chosen with the primary mission objective in mind"):
> "As a concrete example, an over-weighted momentum-management term yields a feasible, converged trajectory
> that nonetheless desaturates in a single body-torquing impulse---a multi-degree pointing transient---whereas
> a lighter weight lets the wheel hold a constant offset and preserves sub-degree pointing. The solve
> converged in both cases; the difference is entirely in the plan, which is why closed-loop quality tracks
> plan quality rather than tracker tuning."

---

## Paper 1 (Allocation)

### 4. Connect LP-vs-QP direction error to the controllability geometry  **[judgment]**
QP tilts the achieved torque toward the achievable plane ($\perp\mathbf{B}$) exactly when the request has a
component along $\mathbf{B}$ — i.e., the direction error is largest where the torque polytope is *thinnest*,
which is the same axis the controllability analysis (\S\ref{sec:controllability}, `tab:ranks`) flags as
field-limited. One sentence unifies the paper's two headline results.

**Draft sentence** (\S\ref{subsec:allocation}, after the QP direction-error discussion):
> "This direction error is not arbitrary: it is largest precisely along the instantaneous field direction,
> the same axis the controllability analysis of \S\ref{sec:controllability} identifies as unachievable for a
> magnetorquer. QP's worst case coincides with the thinnest dimension of the achievable torque polytope."

### 5. The RW-failure "bounded, not tumbling" claim needs the window caveat  **[verified, Task 4]**
Boundedness is demonstrated over a 1000 s window, which is **shorter than the ~5800 s orbit** — the
uncontrollable (instantaneous-$\mathbf{B}$) axis has not swept a full revolution, so the bounded excursion is
a single sweep, not a proven long-horizon bound. The body-rate turnover (peak then decay) is the cleaner
"not tumbling" evidence. State the scope or run multi-orbit (see `FAILURE_RESULTS.md`).

**Draft sentence** (failure subsection):
> "The degradation is bounded over the demonstrated horizon: the full-attitude error peaks and turns over
> rather than diverging, and the body rate peaks below $0.4^\circ$/s and decays. Because this horizon is
> shorter than one orbit, this shows a single bounded excursion as the uncontrollable axis rotates away; a
> multi-orbit run would be required to claim long-horizon boundedness."

### 6. Scope the field-model robustness claim correctly  **[verified, BFIELD_NOTE]**
The nominal / twin-sensitivity campaigns run on the OldPlanner, which is **fed the plant's field** — there is
no plant/planner field mismatch on that path. The real field-error robustness comes from the **deliberate
B-error injection test** and the **SALTRO/P2.6 path** (~$4.3^\circ$/$4.4\%$). Do not present the nominal twin
sweep as evidence of field-model robustness.

---

## Both papers

### 7. The flagship 16% (3+0 reduced) — failure mode CONFIRMED (and my first guess was wrong)  **[verified]**
I tested the obvious hypothesis ("goals near the uncontrollable axis → blocked during the slew") on
`mc100_altro_3+0_reduced` and **it is refuted**: slew-phase blocking is identical for converged vs. failed
(0.48 vs 0.48), and failed trials did not start with harder slews (init ~54° both; one failed trial started
only 12° away). The 16% actually splits into **two populations**:
1. **Horizon truncation (~4–6 trials):** strongly negative end-slope (−90 to −194 °/1000 s), large finals
   (69°, 89°) — still slewing when the 1000 s cutoff hit; would converge with a longer horizon.
2. **Terminal MTQ-hold limit (~10–12 trials):** flat/positive end-slope, finals 5–27°, and the **terminal
   correction axis sits closer to B (0.67 vs 0.53)** — nulling the last few degrees needs a torque near the
   instantaneous field direction, which a magnetorquer cannot produce, so it drifts/limit-cycles instead of
   holding.

**Accurate paper sentence** (replaces the "near the uncontrollable axis" framing):
> "The residual non-convergences split between trajectories still slewing at the $1000$ s cutoff and
> trajectories whose terminal residual requires a torque near the instantaneous field direction---the
> magnetorquer-only hold limit---producing a small drift rather than a clean hold, rather than a failure to
> acquire."

(Diagnostic: `fig_mtq_failure_modes` / `MTQ_16PCT_FAILURE_NOTE.md`.)

### 8. Foreground that reduced-attitude + foresight are *multiplicative*  **[judgment]**
The underactuated-feasibility case rests on the 2-DOF (reduced-attitude) relaxation **combined with** planning
over the time-varying field. Both papers state the pieces separately; neither says the 84% MTQ-only result
needs both at once — the free axis only helps because the planner can place it across a whole orbit of
changing geometry.

### 9. Keep to paired / relative metrics across testbeds  **[framing lock]**
Avoid any absolute convergence-% comparison across different testbeds; keep planner-vs-baseline comparisons
paired by seed within a single configuration (already the case in `tab:p2matrix` — just guard against it
creeping into prose).

---

### Confidence summary
- **Strongest / verified against code+results:** 1, 2, 5, 6, and the disturbance fact under 1.
- **Editorial judgment:** 4, 7, 8.
- **House style:** 3 (example), 9 (framing).
