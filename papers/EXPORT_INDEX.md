# SmallSat/JGCD mega-paper export — 2026-06-04

Figures, data, and info for both papers, plus suggested `.tex` changes
(`TEX_CHANGES_SUGGESTED.md`). The `.tex` manuscripts were **not** edited.

## What changed this session (2026-06-04)
- **NEW test** `Paper2_Planner/fig_disturbance_robustness.*` + `DISTURBANCE_ROBUSTNESS_RESULTS.md` —
  backs the abstract's "disturbances excluded from the planner's model": real residual-dipole (1.1 µN·m,
  dominant) + drag + SRP (via cp–cg offset) + GG in the plant, planner models none of it → convergence
  **100%→100%**, mean final **0.11°→0.22°** (3+1 reduced, N=20 paired). See REVIEW_NOTES §1.
- **NEW diagnostic** `Paper2_Planner/fig_mtq_failure_modes.*` + `MTQ_16PCT_FAILURE_NOTE.md` — the 3+0
  reduced 16% non-convergence, confirmed and split into horizon-truncation (5) + terminal MTQ-hold limit
  (11); refutes the naive "goal near uncontrollable axis" guess. See REVIEW_NOTES §7.
- **NEW** `Paper2_Planner/fig_itworks_planner_vs_pd.{png,pdf}` — grouped-bar "It Works"
  figure (restates `tab:p2matrix`; optional addition, see TEX_CHANGES §4).
- **NEW** `Paper2_Planner/fig_spin_pe.png`, `fig_spin_omega.png`, `fig_spin_h.png` — the spin
  trio the manuscript references, from the saved ADCS_wt run (179°→0, ω_z→~13°/s, h-margin 1.70).
- **LINEAR** regen: `fig_graceful_3mtq0rw`, `fig_mismatch` (Paper 2); `fig_difflaw_mc` (Paper 1,
  re-run at 100-trial scale, reproduces the committed table exactly).
- `fig_lp_precession_failure` confirmed already linear; `fig_same_pd` stays log (per brief).
- **INFO** `NEEDVAL_FILL.md` — every `tab:sim_params_p2` `\needval` answered; the 4 previously-open
  items resolved (nominal J corrected to BeaverCube diag(0.0314,0.0341,0.0100); PD gains; planner
  field; disturbances = GG ~10⁻⁸ N·m, drag/SRP zero-torque).
- `SIM_PARAMS.md`, `BFIELD_NOTE.md` updated.

## Layout
- `Paper1_Generalized_ACS/` — allocation paper figures, tables, RESULTS/info.
- `Paper2_Planner/` — planner paper figures, tables, RESULTS/info.
- `TEX_CHANGES_SUGGESTED.md` — suggested manuscript edits (review + paste; not applied).
- `_pubstyle.py` — shared IEEE figure style helper (used by fig_itworks).

## Known mismatch to resolve
`generalized_acs.tex` line 626 includes `fig_torque_polytope.png`; the shipped file is
`fig_polytope.png` (kept as-is per request). See TEX_CHANGES §5.
