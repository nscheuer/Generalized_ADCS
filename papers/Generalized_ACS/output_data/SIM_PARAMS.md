# Task 2 — Simulation parameters from the repo config

Values read from the actual campaign/config code (file:line provenance). **Flag back** items at the
bottom are genuinely not pinned in the code — leave as `\needval{}`. P2.6 (ESPA, J=diag(1,1.2,0.7),
rods 50 A·m², RW h_max 0.5 N·m·s) is already in the paper and omitted here.

## Actuators & inertia

| Quantity | Paper 1 (LP/QP allocation) | Paper 2 nominal (planner-vs-PD, multigoal, twin, spin) | Source |
|---|---|---|---|
| Inertia J | diag([0.022, 0.022, 0.004]) kg·m² | **diag([0.0314, 0.0341, 0.0100]) kg·m²** (BeaverCube1/2; spin demo overrides to diag(0.10,0.05,0.005)) | P1: `_paper1_sim.py:115`. P2: `_paper2_sim.py:26-30` → `create_beavercube{1,2}_cubesat` |
| MTQ max dipole | 0.2 A·m² | 0.2 A·m² (ISIS board) | `create_cubesat_MTQ.py:23`; `closed_loop_comparison.py:288` |
| RW max torque | **0.001 N·m** | **0.0023 N·m** (CubeWheel SmallPlus) | `closed_loop_comparison.py:289`; `create_cubesat_RW.py:24` |
| RW max momentum h_max | **16.2e-3 N·m·s** (BeaverCube ref) | **0.0036 N·m·s** | `lp_qp_cqp_campaign.py:88`; `create_cubesat_RW.py:27` |
| RW rotor inertia J_rw | 0.001 kg·m² | 5.7e-6 kg·m² | `closed_loop_comparison.py:291`; `create_cubesat_RW.py:25` |

NOTE: Paper 1 and Paper 2-nominal are **different reaction wheels** (P1 uses the BeaverCube
0.001 N·m / 16.2 mN·m·s reference via the allocation `ClosedLoopConfig`; P2 nominal uses the
CubeWheel SmallPlus 0.0023 N·m / 3.6 mN·m·s factory part). Same MTQ and same inertia.

## Control / timestep
| | Paper 1 | Paper 2 nominal |
|---|---|---|
| kp, kd | 5e-5, 1e-3 | (planner: ALTRO cost weights; PD baseline kp/kd — see `_paper2_sim.make_planner_settings`) |
| t_f | 1000 s | 1000 s (1500 s for multigoal P2.3) |
| dt (sim / control) | 2 s | 1 s |
| planner | n/a (allocation only) | `Plan_and_Track_LQR` (ALTRO + TVLQR) |

## Orbit
- Circular, **7000 km radius**; R = 7000·[0, √2/2, √2/2] km, V = [8, 0, 0] km/s → **inclination ≈ 45°**, period ≈ 5800 s. Source: `_paper1_sim.py:83-84`, `_paper2_sim.py:101-104`.
- **Paper 1: orbit is FIXED** (only attitude/rate/momentum randomized).
- **Paper 2 multigoal: orbit is RANDOMIZED** via `create_random_circular_os(radius_km=7000, ...)` (`generate_p2.3_multigoal.py`). Exact randomized elements + ranges → **FLAG** (see below).

## Monte-Carlo randomization
**Paper 1** (`lp_qp_cqp_campaign.py:185-218`): 100 trials, seeds `range(1000,1100)`, convergence < 5°.
- Initial attitude: random axis, rotation angle ~ U(0.1, 0.5) rad.
- Initial body rate: N(0,1)³ × 0.02 rad/s.
- Initial RW momentum: magnitude ∈ {0, 0.3, 0.6, 0.9} × 16.2e-3 N·m·s; random body direction (3+3) / sign-randomized scalar (3+1). (Paired across the four momentum levels by seed.)

**Paper 2 multigoal** (`generate_p2.3_multigoal.py:72-97`):
- Initial body rate: uniform random direction × U(0.1, 1.0) °/s.
- Initial attitude: random unit quaternion.
- Initial RW momentum: U(-1e-4, 1e-4) N·m·s per wheel.
- Orbit: random circular at 7000 km.

## Plant disturbances
- **Paper 2 nominal** (`create_cubesats.py:85-87`): gravity-gradient + atmospheric drag (CD=2.2, six geometry faces, areas 0.1×0.3 and 0.1×0.1 m², η_s=0.5/η_d=0.2/η_a=0.3) + SRP, all **enabled**. Orbit propagated with J2.
- **Paper 1** (`closed_loop_comparison.py` `ClosedLoopConfig`): carries J / actuator maps / limits only — GG/drag/SRP **not** attached to the allocation-campaign plant (the only environmental driver is the time-varying B-field). → **confirm/FLAG**.
- Geomagnetic field: IGRF-13 via `ppigrf` + Skyfield frames (plant). See `BFIELD_NOTE.md`.

---
## Flag back — status (updated 2026-06-04; see `papers/Planner/output_data/NEEDVAL_FILL.md`)
1. **Paper 1 plant disturbances** — `ClosedLoopConfig` attaches no GG/drag/SRP; the only environmental driver in the allocation campaign is the time-varying B-field. (Paper 2 plant: GG effective ~10⁻⁸ N·m; drag/SRP enabled but zero-torque, no cp–cg offset.) — **RESOLVED**
2. **Paper 2 multigoal randomized orbital elements + ranges** — `create_random_circular_os(radius_km=7000)`; exact incl./RAAN/phase ranges still not pinned in `orbit_factory.py`. — **still open**
3. **PD baseline gains** — Lovera (3+0): p=1e-4, d=1e-3, eps=1.0; LP-PD (3+1): p=5e-5, d=2e-3, c=1e-3, h_target=0 (`_paper2_sim.py:83-92`). — **RESOLVED**
4. **Per-campaign t_f/dt for spin (P2.2) and twin (P2.8)** — spin run is tf≈500 s / dt as saved (ADCS_wt); twin/mismatch tf=1000 s, dt=1 s. — partially resolved
5. **Paper 2 nominal J** — BeaverCube `diag(0.0314,0.0341,0.0100)`, **not** `diag(0.022,...)` (earlier guess corrected). — **RESOLVED**
