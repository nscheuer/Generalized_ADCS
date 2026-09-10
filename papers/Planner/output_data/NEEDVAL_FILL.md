# `\needval{}` fill-map for `tab:sim_params_p2` (Paper 2 manuscript)

Each `\needval{}` token in the current `tab:sim_params_p2`, paired with the value from the
repo config (provenance in `papers/Generalized_ACS/output_data/SIM_PARAMS.md`). ✅ = fill
directly; ⚠ = partial / confirm. Spin-demo overrides are already inline in the table.

| Table row | `\needval{}` asks | Value to fill | Status |
|---|---|---|---|
| Inertia $J$ (nominal) | kg m² | `diag(0.022, 0.022, 0.004)` kg·m² | ⚠ confirm — `_paper1_sim.py:115` / `create_cubesats.py`; manuscript leaves nominal open while spin demo is `diag(0.10,0.05,0.005)` |
| Magnetorquers | max dipole, A m² | `0.2` A·m² (ISIS board) | ✅ `create_cubesat_MTQ.py:23`; `closed_loop_comparison.py:288` |
| Reaction wheels | max torque, mN m | `2.3` mN·m (= 0.0023 N·m, CubeWheel SmallPlus) | ✅ `create_cubesat_RW.py:24` |
| Reaction wheels | max momentum, mN m s | `3.6` mN·m·s (= 0.0036 N·m·s) | ✅ `create_cubesat_RW.py:27` (spin override $h_{\max}{=}2$ already inline) |
| Field model (planner/twin) | analytic dipole / low-order IGRF; differentiable | **Nominal MC (OldPlanner `Plan_and_Track_LQR`): planner is *fed the plant's* field** (`ppigrf` IGRF-13 + Skyfield) — planner field ≡ plant field, **no mismatch**. SALTRO C++ path (P2.6 only) uses its own Fortran IGRF-13 + GMST-only frame. | ⚠ see `BFIELD_NOTE.md`. FLAG: whether the planner's *internal differentiable* model is the fed B-array (time-varying parameter) or a re-derived analytic field — not isolated in code. |
| Field model (plant/sim) | IGRF order/epoch | IGRF-13 via `ppigrf` + Skyfield GCRS/ITRS frames | ✅ `ADCS/orbits/orbit.py` |
| Disturbances (MC) | modeled in plant, excluded from planner; magnitudes | Gravity-gradient + atmospheric drag (C_D = 2.2) + SRP, **all enabled** in plant, **excluded** from the planner model; orbit propagated with J2 | ⚠ `create_cubesats.py:85-87` for the *models*; per-axis torque **magnitudes** not tabulated — FLAG if a number is needed |
| Control rate $\Delta t$ | confirm MC dt | `1` s (Paper 2 nominal; Paper 1 uses 2 s) | ✅ `_paper2_sim.py` |

## Previously-open items — now RESOLVED (read out of the code, 2026-06-04)

1. **Nominal $J$** — ⚠ **correction**: the MC does **not** use `diag(0.022,0.022,0.004)`. `_paper2_sim.py:26-30`
   maps the configs to BeaverCube factories: 3+0 = `create_beavercube1_cubesat`, 3+1 =
   `create_beavercube2_cubesat`, 3+3 = `create_3_3_beavercube2_cubesat`. All three carry the same
   inertia: **$J=\mathrm{diag}(0.0314,\,0.0341,\,0.0100)$ kg·m²** (measured from the instantiated sat).
   (The spin demo's `diag(0.10,0.05,0.005)` is a separate custom override, already inline in the table.)

2. **Planner internal field representation** — `Plan_and_Track_LQR` (OldPlanner) is **fed the plant's
   sampled `ppigrf` IGRF-13 + Skyfield field** as a per-timestep parameter $\mathbf{B}(t_k)$; there is **no
   separate analytic dipole**. The magnetorquer torque $\mathbf{m}\times\mathbf{B}$ is *linear in the
   control*, so $\partial\boldsymbol{\tau}/\partial u$ is analytic with $\mathbf{B}(t_k)$ held fixed — that
   is the sense in which it is "differentiable for analytic Jacobians" (the field itself is not
   differentiated w.r.t. state). So planner field ≡ plant field on this path (no mismatch); see `BFIELD_NOTE.md`.

3. **Disturbance torque magnitudes** (plant, BeaverCube2 3+1, over one 5800 s orbit) —
   **Gravity-gradient is the only effective disturbance: peak $4.3\times10^{-8}$, mean $2.4\times10^{-8}$ N·m.**
   Drag and SRP are *enabled but produce identically zero torque* (`tau = -0.0`): the model has **no
   center-of-pressure / center-of-mass offset**, so there is no moment arm (drag is also negligible at the
   ~622 km altitude regardless). So the honest table entry is: GG $\sim\!10^{-8}$ N·m (modeled, excluded
   from planner); drag/SRP enabled but zero-torque (no cp–cg offset).

4. **PD baseline gains** (`_paper2_sim.py:83-92`) — **Lovera (3+0):** `MTQ_Lovera`, $p=1\times10^{-4}$,
   $d=1\times10^{-3}$, $\epsilon=1.0$. **LP-PD (3+1):** `MTQ_w_RW_LP`, $p=5\times10^{-5}$,
   $d=2\times10^{-3}$, $c=1\times10^{-3}$, $h_\mathrm{target}=0$.

## Cross-checks that PASSED against the manuscript
- `tab:p2matrix` cell values (84/27, 18/0, 99/97, 94/90 and the means 3.85/21.54/18.31/85.17/0.26/1.58/1.19/9.06°) reproduce **exactly** from `p2p1_matrix_stats.json`. The new `fig_itworks_planner_vs_pd` is a faithful grouped-bar restatement of this table.
- Boresight `[0,1,0]` (body $+\hat y$) and the 3+1 wheel "on $+\hat y$" are consistent between the manuscript and the `.sim` metadata.
- MPC-vs-TVLQR prose (TVLQR 0.12° / 100%, MPC 13° / 62%, $K^\top K$ surrogate for $S$) matches the P2.4 campaign + the known SALTRO solver limitation.
