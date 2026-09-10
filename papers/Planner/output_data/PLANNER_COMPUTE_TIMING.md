# Task 6 -- planner solve time + memory for `tab:compute_timing` (mostly a FLAG)

**I could not fill `tab:compute_timing` properly.** Two blockers:

1. **Wrong hardware.** The table wants an **Intel i7 laptop** and a **Raspberry Pi 4**. This
   machine is Apple Silicon (`macOS-26.4-arm64`). I cannot produce i7 / RPi4 numbers.
2. **C++ SALTRO not timeable locally.** The table is for the C++ ALTRO solver, but the current
   `saltro_py` build is out of sync with the Python settings scripts:
   `saltro_py.ConstraintConfig` has no `rw_momentum_limit_scale` (set by
   `generate_fig_spin.make_planner_settings`), so `trajOpt` raises before solving. The C++ timing
   needs that settings mismatch fixed first.

## Reference proxy (Python `Plan_and_Track_LQR`, the nominal-campaign planner, ARM Mac)
Not the table's C++ solver and not the table's hardware -- a loose upper-bound reference only.
500 s, 3+1, median of 3 solves:

| Planner | horizon | solve time | peak RSS |
|---|---|---|---|
| Python `Plan_and_Track_LQR` (ARM Mac) | 500 s | **~2.8 s** (min 2.4 s) | ~263 MB (whole process) |

- The dt=1 s and dt=10 s rows came out **identical** (~2.8 s) because this planner plans at a
  **fixed internal `dt_tp=50` s** (10 knots over 500 s) regardless of the tracker `dt`. So the
  Python proxy does **not** reproduce the table's dt=1-vs-dt=10 knot-spacing split -- that split is
  a property of the **C++ SALTRO** knot count (500 vs 50 knots), which is exactly what must be timed
  on the target hardware.
- Peak RSS is whole-process (Python + ADCS + DE-ephemeris); the planner's own working set is much
  smaller. Not a meaningful "planner memory" figure.

## Action to fill the table
1. Fix the `saltro_py` settings mismatch (`rw_momentum_limit_scale`) so `trajOpt` runs.
2. Time the C++ SALTRO 500 s 3+1 solve at dt=1 s (500 knots) and dt=10 s (50 knots) on the actual
   **i7 laptop** and **RPi 4**, recording wall-clock + the solver's own memory (not whole-process RSS).
3. The C++ solver will be far faster than the ~2.8 s Python proxy above; treat the proxy only as a
   sanity ceiling.
