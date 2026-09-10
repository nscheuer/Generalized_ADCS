# Task 3 — Plant vs planner geomagnetic field model

## Bottom line
There **is** a real plant/planner field mismatch, but **only on the SALTRO path** (P2.6 cold-gas
leak). On the OldPlanner path (the nominal Paper-2 campaigns: planner-vs-PD, multigoal,
twin-sensitivity, spin) there is **no** mismatch. So the "every nominal result is a free
robustness datapoint" framing is **not** supported for the OldPlanner campaigns; it *is* true for
the SALTRO-planned P2.6 result.

## The two field paths

| | Plant (Python simulation) | SALTRO planner (C++) | OldPlanner (`Plan_and_Track_LQR`) |
|---|---|---|---|
| IGRF | `ppigrf` IGRF-13 | own Fortran IGRF-13 (`magnetic_model=2`, `trajOpt.cpp:251`) | **fed the plant's `ppigrf` B** |
| ECEF→ECI frame | Skyfield GCRS/ITRS (precession, nutation, Earth rotation) | **GMST-only `rot_z(-gmst)`**, no Skyfield (`SALTRO/src/math/frames.cpp`) | (plant's Skyfield B) |
| Orbit B source | `ADCS/orbits/orbit.py` (always full IGRF; the `fast` flag is *ignored*, orbit.py:25,108) | `saltro_py.generate_orbit` (C++) | `plan_and_track_base._propagate_environment` → same Python `Orbit` |

The OldPlanner does **not** recompute the field — `_propagate_environment` builds a Python `Orbit`
(same `ppigrf` + Skyfield as the plant) and passes the B array into the C++ solver. So OldPlanner B
≡ plant B.

## Quantified discrepancy (SALTRO vs plant, IGRF-13 both, ~one 7000-km orbit)
- **Direction error: mean 4.3°, max 9.9°**
- **Magnitude: ~4.4% lower** (|B_SALTRO|/|B_plant| ≈ 0.956)
- Orbit position difference: **0.08 km** mean — the two propagators agree, so the field gap is
  **not** a position/predicted-orbit effect.
- |B| over the orbit ≈ 14–35 µT.

Because the propagators agree (0.08 km) and both use IGRF-13, the ~4° / ~4% gap is attributable to
(a) the **frame**: SALTRO's GMST-only rotation vs the plant's full Skyfield Earth-orientation, and
(b) the **IGRF implementation/epoch** (the C++ Fortran routine vs `ppigrf`; a ~4% magnitude offset
is larger than secular drift over the J2000+22 yr epoch, so it is most likely an implementation /
reference-radius / coefficient difference, not just epoch). Exact attribution between frame and
coefficients was not isolated; flag if a precise split is needed.

## Caveat / earlier error
An initial check reported ~11° — that used `magnetic_model=0` (not IGRF-13) in `generate_orbit`. The
SALTRO controller uses `orbit_model=1, magnetic_model=2`; with those the gap is ~4.3°/9.9°.

## For the paper
- `tab:sim_params_p2` field rows: **plant** = IGRF-13 via `ppigrf` + Skyfield frames; **SALTRO
  planner** = IGRF-13 (Fortran) + GMST-only frame; **OldPlanner** = uses the plant's field.
- Twin-sensitivity robustness sentence: do **not** claim a free field mismatch — that campaign is
  OldPlanner (no mismatch). The free ~4°/4% field mismatch is genuine for **P2.6 (SALTRO)** and can
  be cited there as an incidental robustness datapoint (the planner held the leak despite a ~4°-off,
  ~4%-low field model).
- The robustness-to-field-error claim for the OldPlanner is backed by a **deliberate** B-field-error
  injection test (run separately), which held up — so the §V-F robustness story stands on the
  intentional sweep, not on a (nonexistent) free nominal mismatch.
