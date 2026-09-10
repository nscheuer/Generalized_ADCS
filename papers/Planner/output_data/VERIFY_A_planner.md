# Planner paper — Wave A verification (2026-06-05)

Each item: answer + file:line. **FLAG**/**MISMATCH**/**CANT-VERIFY** called out explicitly.
Repo: `/Users/patrickmckeen/Documents/Generalized_ADCS`. Paper: `~/Downloads/planner.tex`.

---

## A1 — Replan cadence (cold-gas IV-E) — **paper is WRONG**
Source: [`generate_p2.6_coldgas.py`](../generate_p2.6_coldgas.py) (the cold-gas-leak demo).
- **Integration dt = 1 s** (`:17`, `PLAN_DT`/`DT`).
- **Plan horizon = 600 s** per replan (`:65` `PLAN_HORIZON = 600.0`).
- **Replan cadence = every 500 s**, scheduled at t = 0, 500, 1000, 1500 s (`:64` `REPLAN_TIMES = [0.0, 500.0, 1000.0, 1500.0]`).
- **Leak onset t = 600 s** (`:19`); handled by the *next scheduled* replan (~1000 s), not an immediate trigger.
- **No disturbance-triggered replan** in the cold-gas demo — the schedule is fixed.

**Verdict:** the paper's "1 s receding-horizon replans" conflates the **integration dt (1 s)** with the
**replan interval (500 s)**. Correct statement: *600 s horizon, replanned every 500 s on a fixed schedule,
1 s integration/control step.* (A *separate* demo — the +30° attitude kick in
[`generate_p2.6_replan.py`](../generate_p2.6_replan.py)`:138-145`, `REPLAN_ERR_DEG=10.0` — does use a
state-based trigger; don't confuse the two.)

## A2 — Plan vs Plan+Sim vs MC, per subsection
| Subsection | driving script | nature |
|---|---|---|
| IV "It Works" planner-vs-PD (2×2) | `analyze_p2_matrix.py` (loads `mc100_*.sim`) | **MC, closed-loop sim**, 100/cell |
| IV "It Adapts" multi-goal | `generate_p2.3_multigoal.py` | **MC, closed-loop sim**, 100 |
| IV "It Discovers" **spin** | `generate_fig_spin.py` | **OPEN-LOOP PLAN** (`:266`→`trajOpt`, plots X/U directly; no simulate/rollout) |
| IV "It Recovers" replan | `generate_p2.6_replan.py` (`:152` `solve_ivp`) | plan + TVLQR **closed-loop sim**, single trial |
| IV "Stays Bounded" graceful | `analyze_graceful_3mtq0rw.py` (loads `.sim`) | **MC, closed-loop**, 100 |
| IV-F twin/mismatch | `generate_p2.8_mismatch.py` | **deterministic sweep, NOT MC** — 15 single trajectories (5 J × 5 B × 5 rot), closed-loop |
| IV-G hyperparam | `generate_p2.7_sensitivity.py` | small **MC**, N=6/setting |

**FLAGS:**
- **Spin must be labeled an open-loop plan**, not a closed-loop result. The canonical script plots the
  optimizer's X/U directly.
- **CANT-VERIFY (important):** the paper's actual spin figures are `spin_paper_*` (generated in an
  `ADCS_wt` worktree, copied in this session), and `spin_paper_omega` shows a **noisy** ω trace — the
  signature of a *closed-loop* sim, not a smooth open-loop plan. I could **not locate the exact ADCS_wt
  script** that produced them, so I cannot confirm whether the *paper* spin figure is open- or closed-loop.
  → If you point me at that script I'll classify it definitively.
- **twin/mismatch (IV-F) is NOT Monte Carlo** — if the prose implies MC there, fix it (it's 15 deterministic single-trajectory sweeps).

## A3 — Convergence/success flag semantics
- **Optimizer "converged" = constraint satisfaction**, not goal/pointing tolerance. SALTRO's
  `ALILQRStatus::Converged` fires when **max constraint violation ≤ tol** over all steps
  (`SALTRO/src/optimizer/alilqr.cpp` `max_constraint_violation`, constraints = actuator bounds, rate,
  momentum, keep-out). It does **not** test pointing error.
- **Dynamic feasibility is independent of the flag:** trajectories come from forward rollout of the
  dynamics (shooting), so they satisfy the dynamics by construction even when the optimizer reports
  not-converged. (planner.tex:97 states this.)
- **Paper MC "converged" is a separate post-hoc metric:** final-timestep pointing error < 5°
  (`analyze_p2_matrix.py:26,53`). 
- **Verdict:** the paper's 3-term split — *dynamically feasible* (always, by rollout) / *constraint-satisfying*
  (optimizer flag) / *goal-converged* (<5° pointing, post-hoc) — is **well-grounded and correct**.

## A4 — Multi-goal goal types
All three goals are **reduced-attitude ECI vector-pointing** goals, with `No_Goal` coasts between
(`generate_p2.3_multigoal.py:62-69`): `ECI_Goal(A)` → `No_Goal` → `ECI_Goal(B)` → `No_Goal` →
`ECI_Goal(C)`. `ECI_Goal` subclasses `Vector_Goal` (2-DOF). **No quaternion goals.**

## A5 — LP allocator field — **confirmed actual IGRF**
`MTQ_w_RW_LP` builds its torque map from the **sensed body field** `b_body = M_mtm_read @ sens`
(`ADCS/controller/mtq_w_rw_LP.py:518`), and the sensed field originates from the plant's
**IGRF-13 (`ppigrf`) + Skyfield** orbit field (`ADCS/orbits/orbit.py:212-244`, stored as `os.B`).
Not a fixed/analytic dipole. ✓

## A6 — Physical parameters vs `tab:sim_params_p2`
| Row | Paper | Code (file:line) | Status |
|---|---|---|---|
| Inertia J | `\needval{}` | `diag(0.0314,0.0341,0.0100)` `create_cubesats.py:55-57` | **UNFILLED** in paper |
| MTQ dipole | `\needval{}` | `0.2 A·m²` `create_cubesat_MTQ.py:23` | UNFILLED |
| RW torque | `\needval{}` | `2.3 mN·m` `create_cubesat_RW.py:24` | UNFILLED |
| RW momentum | `\needval{}` | `3.6 mN·m·s` `create_cubesat_RW.py:27` | UNFILLED |
| Orbit | "7000 km **radius**" | `radius_km=7000` `_paper2_sim.py:109` | ✓ correctly "radius" (≈629 km alt) |
| dt | 1 s | `dt=1.0` `_paper2_sim.py:36` | ✓ |
| body rate | U(0.1,1.0)°/s | `generate_p2.3_multigoal.py:75` | ✓ |
| attitude | uniform SO(3) | random unit quat, `:76` | ✓ |
| RW h0 | at rest | `h=0`, `_paper2_sim.py:97` | ✓ |
| orbit randomization | "random incl/RAAN/phase" | `create_random_circular_os`: random unit R + tangent V `orbit_factory.py:67-75` | ✓ **equivalent** (uniformly-random orbit orientation = uniform incl/RAAN/phase; just parametrized via R/V, not Keplerian — NOT a mismatch) |

**FLAG:** the `\needval{}` placeholders are still in the paper — fill from `NEEDVAL_FILL.md`.

## A7 — Trial counts / horizon
| experiment | N | tf | dt | MC? | file:line |
|---|---|---|---|---|---|
| planner-vs-PD 2×2 | **100/cell** | 1000 | 1 | yes | `analyze_p2_matrix.py:57-65` (`mc100_*`) |
| multi-goal | **100** | timeline (~1.5–1.9 ks — **confirm exact**) | 1 | yes | `generate_p2.3_multigoal.py:98-103` |
| disturbance robustness | **20** | 1000 | 1 | yes (paired) | `generate_p2_disturbance_robustness.py:44` *(NEW — not yet in paper)* |
| twin/mismatch | **1/level, 15 total** | 1000 | 1 | **NO (sweep)** | `generate_p2.8_mismatch.py:47-56` |
| hyperparam | **6/setting** | 1000 | 1 | yes (small) | `generate_p2.7_sensitivity.py:124` |

**FLAG:** verify the paper's cited N for each matches (esp. multi-goal N and tf; twin = "15 sweeps", not "MC").

## A8 — MPC (empirical only)
Single-step MPC **was tested and underperformed TVLQR** (100 paired trials, 3+1 reduced, tf=1000, dt=1,
conv<5°): TVLQR **100% / 0.12°** vs MPC **62% / 13°**, MPC ~2.5× more control effort
(`P2.4_tvlqr_vs_mpc_*.json`; `generate_p2.4_tvlqr_vs_mpc.py:52`; `P2.4_RESULTS.md:54-58`). Empirical fact
only — no mechanism asserted.

## A9 — RW saturation in the spin demo
- **Wrong axis:** the body-fixed disturbance is on **+x̂_b**; the 3+1 wheel cannot apply a constant torque
  about a perpendicular axis while holding pointing. (Canonical `generate_fig_spin.py` RW = +z,
  `create_cubesats.py:61`; paper text describes the wheel storing momentum on +ŷ_b — either way ⊥ the +x̂
  disturbance.)
- **Saturates in ~seconds (paper scenario):** the paper gives $\Omega_{\min}=|\tau_d|/h_{\max}=8.6°/s$ with
  $h_{\max}=2$ mN·m·s ⇒ $|\tau_d|\approx0.30$ mN·m. Time-to-saturation $=h_{\max}/|\tau_d|=2/0.30\approx
  \mathbf{6.7\ s}$. ✓ "within seconds."
- **EXACT numbers, from the committed paper script** (`Nscheuer/SALTRO`, branch `autonomous-spin-backup`,
  `tests/debug/optimizer/alilqr_python/autonomous_spin.py`, "Table 7.3"): $J=\mathrm{diag}(0.1,0.05,0.005)$,
  **RW on $+\hat y$**, RW max torque $2\times10^{-4}$ N·m, **$h_{\max}=2$ mN·m·s** (`:118-132`); prop
  disturbance $\boldsymbol{\tau}_d=[3\times10^{-4},0,0]$ N·m = **0.3 mN·m on $+\hat x$** (`:119`); boresight
  $+\hat z$. The script plans via **`S.trajOpt(...)`** (`:215`) and plots X/U → **open-loop plan** (confirms
  A2 for the actual paper figure). Steady spin $\langle\omega_z\rangle=13.8°$/s = 1.6× the 8.6°/s minimum.
  - **Wrong axis:** prop is on $+\hat x$, wheel on $+\hat y$ — a single wheel exchanges momentum only about
    its own axis, so it cannot produce a sustained torque about $+\hat x$ to cancel the prop while pointing.
  - **Time-to-saturation $= h_{\max}/|\boldsymbol{\tau}_d| = 2\times10^{-3}/3\times10^{-4} = \mathbf{6.7\ s}$.** ✓
- The canonical *local* `generate_fig_spin.py` is **stale** ($\tau_d=40\,\mu$N·m, $h_{\max}=3.6$ mN·m·s,
  leftover `axhline(16.2)`) — do not quote it; use `autonomous_spin.py` Table 7.3 above.
