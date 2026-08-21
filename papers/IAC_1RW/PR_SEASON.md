# PR season — upstream items harvested from the IAC-1RW campaign

General fixes go upstream (nscheuer/Generalized_ADCS or SALTRO); paper-specific stays on
`paper/iac-1rw`. Already merged upstream during the campaign: #119 (star-tracker
availability/anisotropy), #120 (controller sens.axis + UAKF sigma-point NaN).

## 1. Solver-internal time/iteration budget (SALTRO / OldPlanner) — NEW, highest value

The wall-clock timeout wrapping `calculate_trajectory` (SIGALRM at the Python layer) cannot
interrupt the compiled solver: signals fire only between Python bytecodes, so a pathological
C++ grind overruns silently until internal iteration caps bind. Observed 2026-08-19: 18
workers wedged for hours planning full-attitude maneuvers on the magnetorquer-only bus —
the recorded pathological case — with the timeout inert exactly where it was needed.

This is a silent failure of the fallback *hierarchy*, not just an inconvenience: on orbit, a
solve that overruns without the timeout firing means the spacecraft rides the overlap window
and then falls through to reactive control with no diagnostic. The planner architecture's
"fallback catches non-convergence" claim rests on the trigger firing.

Fix direction: enforce the budget inside the solver loop — an iteration-count budget and/or
a per-iteration wall-clock check with clean early-return (best-so-far trajectory + status
flag), so the Python layer's timeout becomes a backstop rather than the only line. The paper
carries a limitations sentence on this (PASTE_timeout_limitation.md).

## 2. `_rw_hdot_kernel` double-documentation (sign convention at the interface)

The kernel docstring documents wheel-internal torque; the command is body torque (ḣ = −u,
settled by 3-step empirical probe). The clamp was specified from the documented convention
and inverted the physical one — pumped the wheel to 83 h_max before the negative test caught
it. Upstream fix: rename or double-document the kernel so the body/wheel-internal convention
is unmissable at the call site. The framework thesis (laws port across the interface) cannot
survive a sign ambiguity *at* the interface. Method note: when documentation and physics can
disagree, ask the integrator.

## 3. Step-aware wheel saturation envelope (allocator-level)

`enforce_wheel_envelope`: u ∈ [(h − h_max)/dt, (h + h_max)/dt] per wheel — boundary-only
checks overshoot mid-step (τ_w·dt was 13% of h_max on the settled bus). Empirically-signed
(see item 2). Belongs at the allocation layer, not the integrator, so planning and control
see the same physical box. Include the negative tests (envelope must bind when commanded
past the limit; must not bind inside it) — three consecutive bugs were caught only by them.

## 4. ReservedDesatLP as a probe for the genACS open question

Desat-first allocation (pay the despin channel, pointing gets the remainder). Adjudicated
2/11 conversions on the diverged set → priority allocation does not rescue the high-σ
regime; the frontier is architecture, not allocator pricing. Worth upstreaming as an option
+ the negative result in its docstring: it closes genACS's open question ("would priority
allocation help?") with a measured no for this class of draw.

## 5. Windowed replanning recipe (if the planner stack wants it)

PLAN_WINDOW_S/PLAN_OVERLAP_S structure with feedforward fallback, from `_iac_sim.py`.
Port only after item 1 lands — the recipe inherits the timeout limitation until then.

## 1b. Worker-aging degradation (companion to item 1, measured 2026-08-21)

The solver library accumulates process-level state across planner-object lifetimes:
on a worker that has run a few trials, solves degrade from ~2 s to wall-budget kills
(300 s+) and previously-clean problems fail; the same problems run clean on fresh
workers (crossover, both arms). Nondeterminism is state-dependent, not intrinsic --
fresh workers reproduce bitwise on most draws. Upstream: (a) MonteCarloRunner
max_tasks_per_child pass-through (implemented on paper/iac-1rw, 2-line change);
(b) a state audit of the .so's statics for the real fix. Interaction note: only
full-attitude solves crossed the kill threshold on aged workers; reduced solves
merely slowed.
