# One Wheel Is Enough — campaign spec (revision 2)

**Paper:** IAC-26,B4,6A,2,x109468 · manuscript due 14 September 2026
**Branch:** `paper/iac-1rw` = `origin/main` + 4 cherry-picks · **worktree `ADCS_wt/iac-1rw`**

This supersedes revision 1. It records only what **changed** and why; anything not listed here
stands as originally specified. The full audit that produced these changes is
`ADCS_wt/IAC_1RW_CAMPAIGN_AUDIT.md`.

> Audit against **this worktree**, not `~/Documents/Generalized_ADCS` — that tree is dirty on
> `paper2-p26-saltro-leak` and lags `main`. Reading it produced one retracted finding.

---

## 0. Resolved decisions

| # | Decision | Value |
|---|---|---|
| D1 | Repo base | branch off `main`; **not** the draft pipeline PR |
| D2 | Campaign E accuracy axis | **sustained** — θ at which availability `A(θ) = 0.95` |
| D3 | Horizon | one orbit for the frontier; **multi-orbit for E's saturation points** (see §4) |
| D4 | Residual dipole | **0.05 A·m²**; 0.1 retained as a labelled sensitivity |
| D5 | Campaign R horizon | each trial runs to **its own** orbital period (genACS orbits are random-altitude) |
| — | Estimator | **UAKF** (there is no MEKF in the repo — see §5) |
| — | Allocator | pinned to **LP** (`MTQ_w_RW_LP`) everywhere; `qpc`/`qpg` never used |

---

## 1. Reference bus — now a factory, not a prose table

`ADCS.satellite_factory.create_iac_6u_bus()`. One definition, every campaign.

```python
from ADCS.satellite_factory import create_iac_6u_bus

sat  = create_iac_6u_bus(n_rw=1)                       # reference: wheel on +z (boresight)
mtq  = create_iac_6u_bus(n_rw=0)                       # 3MTQ+0RW
three= create_iac_6u_bus(n_rw=3)                       # 3MTQ+3RW upper bound
est  = create_iac_6u_bus(n_rw=1, estimated=True)       # the filter's model
```

Verified on construction: `J_COM = diag(0.13, 0.10, 0.05)` exactly, `COM = [0.02, 0, 0]`,
mass 12 kg, `m_max = 0.2`, `τ_w = 2 mN·m`, `h_max = 15 mN·m·s`, tach σ = 1% of `h_max`,
dipole in the augmented estimator state (`dist_param_len == 3`).

**Two things the prose spec got structurally wrong, now fixed in code:**

### 1.1 The cp–cg offset is not a refinement — it is the entire drag and SRP torque

For a uniform box every face satisfies `A_i·d_i = V/2`, so the faceted drag sum
`w = Σ A_i C_D (n̂_i·V)₊ r_i` is **parallel to V** and the kernel's `w × V` vanishes. With `COM`
at the geometric centre the drag torque is **identically zero at every attitude** — measured
1.75e-22 N·m over 2000 random ram directions. The same cancellation applies analytically to SRP
(the specular/diffuse terms give `n̂ × n̂ = 0`; the incident term collapses to `ŝ × ŝ = 0`).

The existing `create_beavercube*` factories all have `COM = np.zeros(3)`. Anyone who copies that
pattern gets a campaign with two of the four disturbances silently at zero. `create_iac_6u_bus`
defaults `com_offset_m = 0.02`.

Note also that the projected area is **already attitude-dependent** in the repo's faceted model —
`A_eff = Σ A_i (n̂_i·v̂)₊` spans **0.021–0.070 m²** (median 0.057) for this box, against the
prose spec's fixed 0.06 m². The fixed number is fine as a hand-calc check and is not what the
simulation uses. With the offset in place the torque is
`τ = ½ρ C_D A_eff(q) |V|² |c| (ĉ × V̂)` — magnitude *and* direction attitude-dependent, which is
exactly what §IV-A rests on: ram-locked ⇒ V̂ body-fixed ⇒ body-fixed secular torque ⇒ wheel
momentum ramps linearly; inertial hold ⇒ V̂ sweeps the body frame ⇒ largely averages out.

### 1.2 The inertia is specified about the COM; `Satellite` wants it about the origin

`Satellite.update_J` applies `J_COM = J_0 − m(‖r‖²I − r rᵀ)`. The factory back-solves `J_0` so
the **dynamics** see the paper's `diag(0.13, 0.10, 0.05)`. Handing `J_0 = diag(0.13, 0.10, 0.05)`
directly would give the dynamics `diag(0.13, 0.0952, 0.0452)` — a 5–10% error on two axes.

---

## 2. Campaign R — three differences, not one

**The spec said "change only the horizon." Three things differ, all verified in `_paper1_sim.py`.**

1. **The genACS wheel is on `+x`, perpendicular to the boresight.** `build_actuators` takes
   `[RW(axis=j) for j in MathConstants.unitvecs][:1]` (`:61-64`) and `unitvecs[0] = [1,0,0]`,
   while `boresight = [0,0,1]`. The IAC reference bus mounts the wheel **along** the boresight.
   `create_beavercube2_cubesat` is also wheel-⊥-boresight (+z wheel, +y boresight), so *both*
   companion papers are perpendicular and **this paper is the first wheel-∥-boresight study.**
   That is a genuinely new configuration and is what Campaign D exists to justify — but it means
   **nothing quantitative transfers from either companion paper**, and R differs from A in wheel
   geometry as well as horizon. R keeps `+x`.

2. **The genACS orbit is randomized per trial** — altitude ~ U(400, 1000) km with random
   inclination/RAAN (`:82-84`). "One orbit" is therefore per-trial (periods 5554–6307 s); D5
   fixes each trial to its own period. `output_data/SIM_PARAMS.md` claiming the orbit is FIXED
   is **stale** — trust the code.

3. **genACS has no disturbances and no estimator.** `Satellite(...)` is built with no
   `disturbances=` argument, sensors are magnetometers only, and control runs on truth state.
   Adding the IAC disturbance set "because it's the reference environment" destroys the
   reconciliation.

**And the number itself is ambiguous.** Three defensible values exist:

| Source | Value | Horizon |
|---|---|---|
| `tab_same_pd.csv` (P1.2, final) | **42%** | 1000 s | ← the spec's number |
| `P1.2_RESULTS.md` prose | 38% | 1000 s | stale, pre-regeneration |
| `tab_difflaw_mc.csv` (P1.3, final) | **49%** | **2000 s** | ← what the SSC26 poster prints |

A reader holding the poster sees 49%. **The footnote must name the table, not just the paper.**
(Separately worth fixing the stale 38% in `P1.2_RESULTS.md`.)

genACS actuator scale for the record: `m_max = 0.4 A·m²`, `τ_w = 7 mN·m`, `h_max = 16.2 mN·m·s`
— a 3.5× stronger wheel than the IAC bus at similar storage (time constant 2.3 s vs 7.5 s). "Do
not grow the actuators" stands; note in the footnote that R therefore runs a *stronger* wheel
than the paper's own reference bus.

---

## 3. Campaign B — define equal-peak-torque, and classify the slew axes

**The `m_max ≈ 67 A·m²` figure is under-defined by 1.4–1.6×.** Three magnetorquers with a *box*
constraint `|m_i| ≤ M` do not have peak torque `M·B`. The max over the box is at a vertex
`m = M(±1,±1,±1)`, giving `|m × B| = M·B·√(3 − (ŝ·B̂)²)` — between **√2·M·B** (B along a body
axis) and **√(8/3)·M·B ≈ 1.63·M·B** (B along the body diagonal). So "equal peak torque" means
`m_max ≈ 41–47`, not 67, under the box reading.

**Adopted definition:** `τ_MTQ^max(B) ≜ max_{‖m‖∞ ≤ M} ‖m × B‖`; choose `M` so the **orbit
median** of that equals `τ_w = 2.0 mN·m`. Report the resulting `M` and its min/max over the
orbit. The "physically absurd, this is an ablation not a proposal" label stays and becomes
quantitatively precise.

**Slew-axis classification is required.** At this `m_max` the ⊥B directions have 1.4–1.6× *more*
torque than the wheel, so "several random axes" will be dominated by fast ⊥B slews and the fit
will flatten toward 1/2. Classify each axis by `|ê_slew · B̂|` at slew start (and check B̂ hasn't
rotated much over the slew), then **fit the two families separately** — along-field as the
headline, cross-field as the control.

**Extend the Θ sweep downward.** The 1/3 exponent is an asymptotic small-time (ball-box)
statement; Θ ∈ {0.2 … 3.0} rad is not that regime, and at Θ = 3 rad on a magnetic-only bus the
slew time approaches the field-rotation timescale and the power law will break. Add **Θ = 0.05
and 0.1 rad**. Expect a clean slope at the low-Θ end and curvature at the high end — predict it
rather than be surprised by it.

---

## 4. Campaign E — one authority scale, and the saturation line needs multi-orbit

### 4.1 The sweep must include `h_max`

The spec sweeps `m_max` and `τ_w` down together. The storage side of the saturation condition is
`τ_sec · T_orbit ≤ h_max`, which contains **neither**. As specified the x-axis moves the agility
and drift boundaries while **the saturation boundary stays exactly where it is.**

Fixed in code: `create_iac_6u_bus(authority_scale=s)` scales `m_max`, `τ_w` **and** `h_max`
together, preserving the 7.5 s wheel time constant (a real hardware invariant). Verified:
`s = 0.4` → `m_max = 0.08`, `τ_w = 0.8 mN·m`, `h_max = 6 mN·m·s`, time constant 7.500 s.

### 4.2 One orbit is structurally blind to saturation

With `h_max = 15 mN·m·s` and a secular drag torque ~0.2 μN·m, per-orbit accumulation is
`0.2e-6 × 5554 ≈ 1.1 mN·m·s ≈ 7% of h_max`. Saturation from rest takes **5–15 orbits**. Since
wheels start at rest everywhere except Campaign C, **no cell in A and no cell in E at a
one-orbit horizon can reach the saturation boundary.**

Fine for A — momentum is reported as peak/final fraction of `h_max`, which is the right thing to
report at 7%. Not fine for E. Do both:

- run E's saturation-line points **multi-orbit (5–10)** — there are only five or six of them;
- compute the saturation line **analytically** from D's measured per-orbit dump capacity and the
  measured per-orbit accumulation, using closed-loop runs to confirm two points. The spec already
  takes this approach for F.

Campaign C (h₀/h_max up to 0.75) is the only place saturation is reachable in one orbit — it is
doing double duty and should be described that way.

### 4.3 Accuracy axis

`A(θ)` = fraction of the held interval (post-acquisition) with boresight error ≤ θ; the axis is
the **θ at which `A(θ) = 0.95`**. Final error is an artifact of where the replanning window
boundary lands — executing to a plan endpoint produces window-joint spikes from shrinking TVLQR
gains. This definition is already implemented in `pointing_availability/`.

### 4.4 Cost

As specified (2 controllers × 2 panels × boundary sampling × 100 trials) E is plausibly 5× A and
half planner runs. **Use 50 trials for E**, reuse A's twelve cells as free anchor points, and
bracket each predicted boundary with 5–6 points. Paired seeds mean 50 trials still resolves the
planner-vs-PD envelope difference, which is a paired quantity and is the headline.

---

## 5. Estimation

**There is no MEKF in the repo** — only `UAKF` and `SRUAKF`. The augmented-state architecture
(gyro bias + sensor bias + disturbance params) is already there, so this is a naming problem, not
a capability problem, **but the paper must not say MEKF.** Use `UAKF` and cite it as a UKF.
(`ESTIMATOR_RESTRUCTURE_BRIEF.md` records four SRUAKF drifts vs the UAKF; UAKF is also the safer
choice on the merits.)

**Residual-dipole estimation works out of the box.** `Dipole_Disturbance` has `main_param`,
`torque_valjac = [B×]`, `torque_qvalhess`, `torque_valvalhess`, and matching call signatures;
`testing/test_estimators/test_disturbance_param_estimation.py` passes 8/8. Use
`Dipole_Disturbance(estimate_dist=True)`, **not** `General_Disturbance` — the latter estimates a
constant **τ**, and cancelling a residual dipole requires **m_res**, because `m_cmd = −m_res`
tracks the field automatically while an inverted τ is correct at one instant only.

**Sensor grades** are in `ADCS.satellite_factory.sensors.create_iac_*` with both noise
conversions (ARW → per-sample σ; Allan bias instability → random-walk σ) written out in the
module docstring, because the paper has to state them once.

**Field-model error must actually be injected.** `simulate.py:233` already separates
`os_for_gnc` from the plant's `os_k`, so the ~4°/4% error is a wrapper on that path. Without it
estimator and plant share one B and field-model error contributes **zero** to the dipole-estimate
residual — which makes the §IV cancellation result optimistic in exactly the way the spec warns
against.

---

## 6. Residual dipole — two consequences to carry into the writing

- **It is not purely cyclic.** `τ = m_res × B_body` with a body-fixed `m_res` and a nadir-locked
  attitude has a non-zero orbit mean in general, so there is a secular component. Since the dipole
  is the largest disturbance by ~15×, its secular fraction is plausibly the **dominant** momentum
  driver — larger than drag. Measure it in the §2 cross-check rather than assuming.
- **Cancelling it costs `m_res/m_max` of the magnetorquer budget, permanently** — 25% at the
  reference 0.05 A·m², 50% at the 0.1 sensitivity. The magnetorquer duty-fraction metric will
  read ≥ that floor in every cancelling cell. Strong quotable result for §IV.

At 0.1 A·m² the disturbance-to-authority ratio would be ~0.5 and the 3MTQ+0RW cells would fail on
disturbance budget rather than rank — hence 0.05 for the reference bus (ratio ≈ 0.25). Do not
report an MTQ-only failure as a rank result when it is a budget result.

---

## 7. Sequencing

| Step | Work | Status |
|---|---|---|
| 0 | Reference-bus factory, sensor grades, star-tracker availability | **done** — star-tracker half upstreamed as GenADCS PR #119 |
| 1 | §2 environment cross-check (budget vs the four hand numbers; **measure the dipole's secular fraction**; confirm SRP non-zero once COM ≠ 0) | next — gates everything |
| 2 | R — 100 trials × 2 variants, truth-state, no disturbances, `+x` wheel | |
| 3 | A — 12 cells × 100 trials × one orbit, **both horizons from one run** | planner cells need the iteration caps + SIGALRM backstop + reactive fallback; `trajOpt` **raises** on non-convergence |
| 4 | B / C / D in parallel — D is largely a `pointing_availability/test5_sigma_duty.py` | |
| 5 | E — boundary-bracketed, saturation points multi-orbit, 50 trials | |
| 6 | F — 4 altitudes, mostly analytic (uses the cherry-picked solar-activity density) | |

Density is **SMAD table + F10.7 scaling**, not NRLMSISE-00 (which the repo does not have).
Describe it honestly; the difference is invisible at 400 km for a torque budget and very visible
at 800 km for Campaign F, where the two-orders-of-magnitude density claim lives.
