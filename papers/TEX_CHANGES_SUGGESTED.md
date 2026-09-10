# Suggested `.tex` changes (NOT applied — for you to review and paste)

Generated 2026-06-04. These are *suggestions* keyed to the current `planner.tex` and
`generalized_acs.tex`. Nothing in your manuscripts was edited. File/line numbers refer to
the full `.tex` you provided.

---

## Paper 2 — `planner.tex`

### 1. `tab:sim_params_p2` — fill the `\needval{}` tokens
Source + provenance: `Paper2_Planner/NEEDVAL_FILL.md`. Drop-in values:

| Row | Replace `\needval{...}` with |
|---|---|
| Inertia $J$ (nominal) | `$\mathrm{diag}(0.0314,\,0.0341,\,0.0100)$` (BeaverCube1/2; **not** 0.022) |
| Magnetorquers max dipole | `0.2` |
| RW max torque | `2.3` |
| RW max momentum | `3.6` |
| Field model (planner/twin) | `IGRF-13 (ppigrf) + Skyfield, fed to the planner as a per-step parameter $\mathbf{B}(t_k)$; torque $\mathbf{m}\times\mathbf{B}$ is linear in $u$ so Jacobians are analytic` |
| Field model (plant/sim) | `IGRF-13 via ppigrf + Skyfield GCRS/ITRS frames` |
| Disturbances (MC) | `gravity-gradient (peak $4.3\times10^{-8}$, mean $2.4\times10^{-8}$ N·m), excluded from planner model; drag + SRP enabled but zero-torque (no cp–cg offset)` |
| Control rate $\Delta t$ | `1` s (confirmed) |

### 2. PD baseline gains (for `subsec:planner_vs_pd`, if you cite them)
Lovera (3+0): $p=10^{-4}$, $d=10^{-3}$, $\epsilon=1.0$. LP-PD (3+1): $p=5\times10^{-5}$,
$d=2\times10^{-3}$, $c=10^{-3}$, $h_\mathrm{target}=0$. (`_paper2_sim.py:83-92`)

### 3. Spin trio (`fig:spin_pe`, `fig:spin_omega`, `fig:spin_h`) — now provided
`fig_spin_pe.png`, `fig_spin_omega.png`, `fig_spin_h.png` are in `Paper2_Planner/`. They are
the saved run (179°→0; ω_z spins to ~13°/s, transverse damped; h peaks 1.70, settles
~−1.2 mN·m·s). **Captions already match — no change needed.**

### 4. OPTIONAL — add the grouped-bar "It Works" figure
`fig_itworks_planner_vs_pd.{png,pdf}` is a grouped-bar restatement of `tab:p2matrix`
(identical numbers). If you want it alongside the table in `subsec:planner_vs_pd`:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_itworks_planner_vs_pd.pdf}
\caption{Planner vs.\ PD baseline across the $2\times2$ matrix. On each configuration's
controllability-appropriate task (bold) the planner converges where PD does not:
MTQ-only reduced pointing $84\%$ vs.\ $27\%$, and 3+1 full attitude $94\%$ vs.\ $90\%$
but mean $1.2^\circ$ vs.\ $9.1^\circ$. Left: convergence rate; right: mean final error
(log).}
\label{fig:itworks}
\end{figure}
```

---

## Paper 1 — `generalized_acs.tex`

### 5. Polytope figure filename mismatch (line 626)
`\includegraphics{fig_torque_polytope.png}` but the file is **`fig_polytope.png`**. Per your
instruction the file is shipped as-is (not renamed). Either change the include to
`{fig_polytope.png}`, or rename the file your side. (Shipped: `Paper1_Generalized_ACS/fig_polytope.png`.)

### 6. `fig_difflaw_mc` — now linear
Regenerated at the correct 100-trial `paper` scale (reproduces `tab_difflaw_mc` exactly: full
LP-PD $42.0^\circ$/median $14.5^\circ$, RW-idle $111^\circ$). Now linear y, with the two task
panels on independent y-scales (a shared linear axis crushed the vector panel). No caption
change needed unless you described it as log.

### 7. Abstract LP direction error (FROZEN — note only, no change)
Abstract says LP mean **$0.004^\circ$**; the regenerated `tab_lp_vs_qp_direction` gives
**~$0.001^\circ$** per config. You said the abstract is frozen and the difference is
negligible — flagging only so the prose and table don't look inconsistent if a reviewer
compares them. Both support "direction-preserving."

---

## Files shipped that map to manuscript figures

| Manuscript `\includegraphics` | shipped file | note |
|---|---|---|
| `fig_spin_pe/omega/h.png` | same names | spin trio, newly provided |
| `fig_graceful_3mtq0rw.png` | same | now linear |
| `fig_mismatch.png` | same | now linear |
| `fig_difflaw_mc.png` | same | now linear, paper scale |
| `fig_torque_polytope.png` | `fig_polytope.png` | **filename differs** (see #5) |
| `fig_itworks_planner_vs_pd.pdf` | same | optional addition (see #4) |
| all others | same names | unchanged |
