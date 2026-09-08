# One Wheel Is Enough — IAC-26 (IAC-26,B4,6A,2,x109468)

Simulation code, campaign data and figure scripts for the paper. Everything the
manuscript reports was produced from this tree; the scripts below regenerate the
figures and the tabulated statistics from the data in `output_data/`.

## Contents

| file | role |
|---|---|
| `_iac_sim.py` | shared closed-loop harness: reference bus, sensor suite, estimator, disturbance set, controller construction, windowed replanning with a process-boundary solve budget, per-trial metrics |
| `_feedforward.py` | LP allocator with residual-dipole feedforward |
| `_field_error.py`, `test_field_error.py` | magnetic-field model error between plant and GNC |
| `_reserved_desat.py` | LP variant that reserves magnetorquer authority for desaturation (the "desaturation-first" trace) |
| `generate_A_baseline.py` | Campaign A: the 3+0 / 3+1 / 3+3 × boresight / full-attitude × PD / planner grid |
| `generate_B_equal_torque.py`, `generate_B_equal_torque_planner.py` | Campaign B: equal-peak-torque slews, feedback and planner |
| `generate_C_bias.py` | Campaign C: stored-momentum bias sweep |
| `generate_D_sigma_duty.py` | Campaign D: σ = \|â·B̂\| duty and dump capacity by wheel mounting |
| `generate_F_altitude.py` | Campaign F: secular momentum vs altitude, momentum-boundary altitude |
| `generate_R_reconciliation.py` | Campaign R: reconciliation with the earlier generalized-ACS result (uses `papers/Generalized_ACS/_paper1_sim.py`) |
| `tune_planner.py` | planner warm-start / cost-weight study, per-task frozen configurations, the 3+1 planner rerun wave, and the QP-allocator MTQ-only cell |
| `check_environment.py`, `verify_bus.py` | disturbance-environment cross-check and bus verification |
| `fig12_supporting.py`, `fig3_envelope.py`, `fig_grid_dots.py`, `fig_failure_seed8.py`, `fig_style.py` | figures |

The reference bus, sensor suite and metrics live in the package:
`ADCS/satellite_factory/satellites/create_iac_6u.py`,
`ADCS/satellite_factory/sensors/create_iac_sensors.py`, `ADCS/helpers/metrics.py`.

## Environment

Run every script from the repository root with the project's Python environment.
Each script puts the repository root first on `sys.path`, so the `ADCS` package of
this checkout is the one that runs. Campaign runs are long (a 100-trial PD cell is
about two hours on 15 cores; planner cells several times that); run one at a time,
with a fresh worker per trial (`max_tasks_per_child=1`, the default in the generators).
Per-trial results are written atomically from the worker, so an interrupted cell
resumes from disk.

## Figures

| output (`output_data/`) | script | inputs |
|---|---|---|
| `fig1_sigma` | `fig12_supporting.py` | σ(t) from bus geometry; medians checked against `D_sigma_duty_20260818_174554.json` |
| `fig2_altitude` | `fig12_supporting.py` | `F_altitude_20260818_174558.json` |
| `fig3_envelope` (mission screening diagram) | `fig3_envelope.py` | grid cells below; 3+0 demand index from `F_altitude_20260818_174558.json` |
| `fig_grid_dots` | `fig_grid_dots.py` | grid cells below |
| `fig_seed8_threeway`, `fig_seed8_threeway_4panel` | `fig_failure_seed8.py` | `wave/pd_reduced_kp1/pd_reduced_kp1_s0008.pkl`, `seed8_reserved.pkl`, `A_trials/1rw_reduced_planner_seed0008.pkl` |
| `fig_pd_hist` | `fig_failure_seed8.py` | `wave/pd_reduced_kp1/*.pkl` |
| `fig_D_sigma_duty` | `generate_D_sigma_duty.py` | written by the campaign |

## Data

Grid cells (Campaign A, one orbit, seeds 0–99 shared across the 3+1 cells):

| cell | data |
|---|---|
| 3+0 and 3+3, both tasks, PD (n = 100, seeds 0–99) | `A_trials/{0rw,3rw}_{reduced,full}_pd_seed*.pkl`, aggregate `A_baseline_20260907_184702.json` |
| the same four cells at n = 30 (seeds 0–29; earlier run, kept because the figure scripts read it) | `A_baseline_20260818_202627.json` |
| 3+1 boresight, PD (n = 100) | `wave/pd_reduced_kp1/` |
| 3+1 full attitude, PD (n = 100) | `wave/pd_full_kp1/` |
| 3+1 boresight, planner (n = 100) | `A_trials/1rw_reduced_planner_seed*.pkl` |
| 3+1 full attitude, planner, per-task tuned weights (n = 100) | `tune_seed*_wave_planner_full.pkl` |
| 3+1 full attitude, planner, baseline weights (n = 100, same seeds) | `tune_seed*_wave_planner_full_base.pkl` |
| 3+0 boresight under the QP allocator (n = 30, same seeds as the context cell) | `wave/qp_0rw_reduced/` |
| seed 8 under the desaturation-reserving allocator | `seed8_reserved.pkl` |

Campaign outputs: B `B_planner_20260822_113100.json`, `B_topup_alongB_20260822_115023.json`
(planner) and `B_equal_torque_20260805_173523.json` (feedback); C `C_bias_20260819_143616.json`;
D `D_sigma_duty_20260818_174554.json`; F `F_altitude_20260818_174558.json`;
R `R_reconciliation_20260805_152714.json`.

Per-trial pickles hold the time histories (attitude error, wheel momentum, σ, LP scale α,
estimator error, tracker availability) and the per-trial metrics; `fig_grid_dots.py`
prints the grid statistics (median, 25–75 %, fractions below 1° and 5° and above 30°)
when run.

The manuscript source is not in this repository.
