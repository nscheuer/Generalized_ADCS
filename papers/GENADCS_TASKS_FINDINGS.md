# GenADCS brief — findings & conclusions (2026-06-06)

All 8 tasks. Per task: finding, numbers/`.tex`-ready values, and any contradiction with the
current paper. Figures regenerated in `output_data/` with in-plot `P1.x(FIG-…)` titles removed.
**Flags** (discrepancies spotted) are collected at the end.

---

## TASK 1 [P0] — portability ("Same Law, Different Configs")  ✅ NOW on RANDOM LEO orbits
Re-run with **random circular LEO orbits** (altitude U(400,1000) km, random inclination/RAAN; was a
single fixed ~622 km / 45° orbit). `tab:same_pd` / `fig_same_pd` updated:

| Config | Task | Conv% | Mean° | 1−ᾱ | (fixed-orbit was) |
|---|---|---|---|---|---|
| 3MTQ | vector | **1** | 94.2 | 0.998 | (1 / 90.3) |
| 3MTQ+1RW | vector | **80** | 8.8 | 0.228 | (71 / 9.6) |
| 3MTQ+3RW | vector | **100** | 0.01 | 0.000 | (100 / 0.01) |
| 3MTQ | full | **0** | 121.1 | 0.999 | (0 / 118.4) |
| 3MTQ+1RW | full | **42** | 44.4 | 0.549 | (38 / 43.6) |
| 3MTQ+3RW | full | **100** | 0.01 | 0.000 | (100 / 0.01) |

Random LEO **slightly improves** the magnetorquer-assisted cells (3+1 vector 71→80%, full 38→42%) —
random inclination spans lower altitudes where the field is stronger (B∝1/r³), giving the MTQs more
authority on average. The MTQ-only cells are unchanged (1%/0% — the controllability wall), and 3+3 is
unchanged (100%). Reconciliation prose holds: magnetic-only is controllable in principle; what fails is
convergence within the 1000 s horizon (1−ᾱ≈1 ⇒ the LP is saturated/zero — see Task 3). `fig_same_pd`:
no in-plot title, band labeled "mean–95th pct". (IC pairing also fixed: config-independent draws now
precede the per-wheel `h0`, so paired configs share an identical scenario+orbit.)

## TASK 2 [P1] — J-scaling validation  ✅ CLAIM VALIDATED (new experiment)
Same law (LP-PD), same scenarios, 3MTQ+3RW full, 100 trials; gains scaled by `trace(J)/trace(J1)`:

| Inertia | tr J | gain× | Conv% (scaled) | Mean° (scaled) | Conv% (UNSCALED) |
|---|---|---|---|---|---|
| J1 (ref) | 0.048 | 1.0 | 100 | 0.01 | 100 |
| J2 (BeaverCube) | 0.076 | 1.6 | 100 | 0.01 | 100 |
| J3 (20× J1) | 0.960 | 20 | **100** | 0.00 | **1** |

Scaled gains → **invariant** (100% across a 20× inertia range). Unscaled (canonical) gains →
**collapse to 1% / 27°** on the 20× bus. The inertia-scaling is load-bearing → **keep the claim**;
add this table (`tab_jscaling.tex`) + `fig_jscaling`.

## TASK 3 [P2] — LP infeasible-direction behavior  🔴 CONTRADICTS §II-E
Verified in `mtq_w_rw_LP.py:917-962`: the LP solves `A_total u = T·τ̂` (strictly direction-preserving)
and **does NOT project τ̂ onto the achievable subspace**. When the request has any component along B
(outside col(A_mtq), which is ⊥B), `T=0` ⇒ it returns **zero torque, α=0**; solver failure also
returns zeros. So §II-E's *"projects τ̂ onto the achievable subspace and re-solves … rather than
returning zero"* is the **opposite** of the code — it **does** return zero. **Soften/correct §II-E**
(or add the projection to the code if that behavior is desired). [This also explains Task 1's 1−ᾱ≈1
for 3MTQ.]

## TASK 4 [P2] — 1MTQ vector LTV rank  ✅ 2/4 CONFIRMED
LTV (orbit-varying field) rank for 1MTQ+0RW, vector task = **2/4** — field variation does **not**
lift it. Uncontrollable directions confirm the conjecture: the single dipole axis m̂ is never
controllable (m×B ⟂ m̂ for all B), and for the vector task m̂ lands in the transverse (⟂â) subspace,
capping the rank at 2. Contrast: 1MTQ Full lifts 2/6→4/6 over the orbit. → keep the one-line
geometric note; `tab:ranks` cell is correct.

## TASK 5 [P2] — MC sampling  ⚠ wording fill (you handle wording)
Code (`_paper1_sim.py`, drives same_pd/difflaw/samelaw/tab_mc): initial attitude = **random
quaternion (uniform SO(3))**; rate = **random dir × U(0.1,2.0)°/s**; **orbit = random circular LEO,
alt U(400,1000) km, random incl/RAAN** (changed this session — was fixed); goal **independent random**;
paired seeds (`seed=run_id`, now config-independent so paired configs share scenario+orbit). Resulting
initial pointing error: **vector ~60° median, full ~132° median**. The "U(0.1,0.5)-from-identity"
phrasing is immaterial here (the independent random goal dominates → identical numbers; verified by
20k-sample sweep). [Detailed fill values were delivered separately.]

## TASK 6 [P1] — planner compute table  🔴 CANNOT FILL (see `Paper2_Planner/PLANNER_COMPUTE_TIMING.md`)
`tab:compute_timing` wants Intel i7 + RPi4 C++-SALTRO numbers; this is ARM Mac and the C++ solver has
a `saltro_py` settings mismatch (`rw_momentum_limit_scale`) blocking a local solve. Reference proxy
(Python `Plan_and_Track_LQR`, 500 s 3+1, ARM): **~2.8 s solve, ~263 MB RSS**; dt=1 and dt=10 identical
(it plans at fixed `dt_tp=50`, so it can't reproduce the table's knot-spacing split). **Action:** fix
the settings mismatch, time C++ SALTRO on the target hardware.

## TASK 7 [P2] — LP/QP allocation timings  ✅ (machine-dependent)
ARM Mac, single process, N=2000, 3MTQ+1RW: **LP 530 µs / QP 62 µs** median (`ALLOC_TIMING_RESULTS.md`).
Paper §V-B cites 350 µs / 40 µs; absolute numbers are machine-dependent, but the **LP:QP ratio ~8.5×
matches** the paper's ~8.75×, and both are sub-millisecond. The single-core desktop caveat in §V-B
stands; flight-processor numbers need direct measurement (ordering holds).

## TASK 8 — figures  ✅ all regenerated, same filenames, no in-plot `FIG-` titles
- **`fig_difflaw_mc` (Fig 4, P1):** Wie **dropped** → 3 laws (LP-PD, Lovera, Wisniewski); enlarged
  (13×5.2); bands lightened (α 0.08) + thin envelope edges + "25–75th pct band" legend entry. (Table
  also now 3 laws; 2 RW-idle baselines kept as dashed in the full panel.) **Now on random LEO orbits**
  — vector conv rose LP-PD 70→83%, Lovera 67→80%, Wisniewski 67→72%; full LP-PD 47→49%, Lovera 22→29%,
  Wisniewski 37→39% (law ordering preserved; Lovera worst on full).
- **`fig_same_pd` (Fig 3):** no in-plot title; shaded area labeled "mean–95th pct band".
- **`fig_failure` (Fig 5):** **REDESIGNED** (replaces the body-fixed-disturbance test). Now a
  **RW failure during full-orbit LVLH (nadir) tracking** on a real orbit (IGRF + gravity-gradient, no
  artificial torque): 3+3→3+2 holds full LVLH (0°); 3+1→3+0 drops to magnetorquer-only, the framework
  relaxes to a **reduced-attitude** goal and **holds boresight→nadir bounded (≤13.5°, settles 3°)** over
  a full orbit while **releasing roll** (full-LVLH grows to ~110°), transverse rate <0.07°/s (not
  tumbling). Ties to Task 3: a *full*-attitude controller on the underactuated bus would feed the
  direction-preserving LP an unachievable torque → α=0 → tumble; projecting out roll keeps it feasible.
  See `FAILURE_RESULTS.md` + `prototype_failure_lvlh.py`. (The old body-fixed + symlog-body-rate version
  is superseded; the ~825 s ω_z-reversal valley it showed was an artifact of that disturbance being
  aligned with the failed axis.)
- **`fig_lp_vs_qp_direction` (Fig 6):** the invisible 3MTQ+0RW LP curve in the CDF now renders (per-
  config colours + circle markers on the near-vertical LP curves + zorder).
- **`fig_lp_precession_failure` (Fig 7):** enlarged (16.5×5.2); suptitle removed.

---

## FLAGS (discrepancies to reconcile in the paper)
1. **§II-E** — LP "project-and-re-solve" is **not** what the code does (returns α=0). [Task 3]
2. **`tab:sim_params_p1`** — code (`_paper1_sim.py:55,115`) uses **J=diag(0.022,0.022,0.004)**,
   **MTQ max=0.4**, **RW max torque=7e-3 N·m** for same_pd/difflaw; the table says J=diag(0.0314…)
   "common to all MC", MTQ 0.2, RW 1.0 mN·m. **These don't match** — the law-swap is **not** on
   BeaverCube J in the current code. Reconcile which is authoritative.
3. **Abstract "randomized orbital parameters"** — **RESOLVED** for the MC studies: same_pd (P1.2) and
   difflaw (P1.3) now use **random circular LEO orbits** (alt U(400,1000) km, random incl/RAAN), so the
   claim is now true for them. The **P1.1 allocation suite** (lp_qp_cqp + `generate_mc_*`) intentionally
   stays on the fixed small-slew testbed (per decision — it's the α-collapse demo, not a random-orbit MC).
4. **MC sampling** — code uses random-quat init + U(0.1,2°)/s rate, not the §IV-A "U(0.1,0.5) from
   identity / N(0,1)³×0.02" (numbers identical regardless). lp_qp_cqp (P1.1) uses a *different* IC
   (identity goal, U(0.1,0.5) → small 13–17° slews) than the 60°/132° MC paragraph implies.
5. **difflaw text (line 809-815, 878)** still says **"four laws" incl. Wie**; the figure/table are now
   **three** (Wie dropped). Update the prose + caption to match.
6. **§V-B timings** — paper 350/40 µs vs measured 530/62 µs (ARM); keep as machine-caveated, ratio holds.
7. **`tab:compute_timing`** — unfilled; needs i7/RPi4 + a `saltro_py` settings fix. [Task 6]

## NOT done (per brief)
Fig 2 (`fig_torque_polytope`) — conceptual redesign, explicitly out of scope.
