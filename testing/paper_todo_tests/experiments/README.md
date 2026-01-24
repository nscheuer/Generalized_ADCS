# Thesis and Paper Experiments

This directory contains experiment scripts for generating all data-based figures in the PhD thesis and 4 associated papers.

## Quick Start

```bash
# List all experiments
python thesis_chapter7_planning.py --list
python thesis_chapter6_disturbance.py --list
python thesis_chapter4_estimation.py --list
python paper_experiments.py --paper 3p1 --list

# Run quick test (10 trials, 100s)
python thesis_chapter7_planning.py --experiment mc_180deg_1rw --quick

# Run full experiment (100 trials, 500s)
python thesis_chapter7_planning.py --experiment mc_180deg_1rw --full
```

## Experiment Scripts

### Thesis Chapters

| Script | Chapter | Description | Est. Runtime (Full) |
|--------|---------|-------------|---------------------|
| `thesis_chapter4_estimation.py` | Ch 4 | USQUE vs DAF filter comparison | 2-4 hours |
| `thesis_chapter6_disturbance.py` | Ch 6 | Wie/Lovera/Wisniewski controllers | 1-2 hours |
| `thesis_chapter7_planning.py` | Ch 7 | ALTRO trajectory planning MC | 8-16 hours |

### Papers

| Script | Paper | Description |
|--------|-------|-------------|
| `paper_experiments.py --paper 3p1` | 3+1 Paper | Architecture comparison |
| `paper_experiments.py --paper generalized` | Generalized Paper | LP vs QP allocation |
| `paper_experiments.py --paper planner` | Planner Paper | ALTRO results |
| `paper_experiments.py --paper package` | Package Paper | Validation tests |

## Thesis Parameters (Extracted from LaTeX)

### Chapter 7: Planning (Table 7.2 - Monte Carlo Satellite)

```
Satellite Inertia:
  J_xx = 0.005256 kg·m²
  J_yy = J_zz = 0.04939 kg·m²

MTQ Limits:
  x-axis: 0.19 Am²
  y/z-axis: 0.57 Am²

Reaction Wheel (y-axis aligned):
  Max torque: 0.0002 Nm (0.2 mNm)
  Momentum storage: 0.002 Nms (2 mNms)
  Wheel inertia: 2e-6 kg·m²

Orbit:
  ISS-like: 51.5° inclination, 429 km altitude
```

### Expected Results (from Thesis Tables)

#### Table 7.3: 180° Slew Monte Carlo
| Metric | MTQ-Only | 3MTQ+1RW |
|--------|----------|----------|
| % within 1° | 11% | 96% |
| % within 10° | 73% | 100% |
| Mean error | 10.2° | 0.11° |
| Median error | 4.2° | 1.2e-4° |

#### Table 7.4: Reduced Attitude Monte Carlo
| Metric | MTQ-Only | 3MTQ+1RW |
|--------|----------|----------|
| % within 1° | 67% | 96% |
| % within 10° | 75% | 100% |
| Mean error | 16.2° | 0.16° |
| Median error | 0.04° | 2.6e-3° |

## Chapter 7 Experiments

### Monte Carlo Tests
- `mc_180deg_mtq`: 180° slew with MTQ-only (Table 7.3)
- `mc_180deg_1rw`: 180° slew with 3MTQ+1RW (Table 7.3)
- `mc_reduced_mtq`: Reduced attitude with MTQ-only (Table 7.4)
- `mc_reduced_1rw`: Reduced attitude with 3MTQ+1RW (Table 7.4)
- `mc_multi_mtq`: Multi-target with MTQ-only (Table 7.5)
- `mc_multi_1rw`: Multi-target with 3MTQ+1RW (Table 7.5)

### Special Tests
- `spinning_solution`: Satellite spinning to counter disturbance (Table 7.1)
- `sequential_planning`: Sequential trajectory planning (Table 7.6)
- `two_goal_trajectory`: Long trajectory with two goals (Table 7.7)

## Chapter 6 Experiments

### Controller Comparisons
- `wie_*`: Wie 3RW controller (Section 6.3)
- `lovera_*`: Lovera MTQ controller (Section 6.4)
- `wisniewski_*`: Wisniewski sliding mode (Section 6.5)
- `wisniewski_twist_*`: Modified Wisniewski (Section 6.6)

## Chapter 4 Experiments

### Filter Comparison Cases
- `case_a`: TRMM, large initial errors
- `case_b`: TRMM, small initial errors
- `case_c`: CubeSat, large initial errors
- `case_d`: CubeSat, small initial errors
- `case_e/f/g`: CubeSat inclusion tests
- `case_g_full`: Full state estimation

## Output Structure

```
experiment_outputs/
├── chapter4_figures/
│   ├── case_a/
│   ├── case_b/
│   ├── inclusion/
│   └── many_var/
├── chapter6_figures/
│   └── (controller comparison plots)
├── chapter7_figures/
│   ├── simple_slew/
│   ├── single_target_imaging/
│   ├── multi_target_imaging/
│   └── sequential/
└── paper_figures/
    ├── 3p1_paper/
    ├── generalized_paper/
    ├── planner_paper/
    └── package_paper/
```

## Key Implementation Notes

### ALTRO Planner Settings
```python
planner_settings = PlannerSettings(
    est_sat=sat,
    bdot_on=0,      # 0=off, 1=simple, 2=advanced
    dt_tp=10,       # MUST be <= 20 for N >= 4 per segment
    dt_tvlqr=1,
)
```

### Satellite Factory Functions
- `create_beavercube1_cubesat()`: MTQ-only (3+0)
- `create_beavercube2_cubesat()`: 3MTQ+1RW (z-axis RW)
- `create_3_3_beavercube2_cubesat()`: 3MTQ+3RW

### Controller API
```python
# MTQ-only
controller = MTQ_Lovera(est_sat, p_gain=0.001, d_gain=0.005, eps=1.0)

# With RW (LP allocation)
controller = MTQ_w_RW_LP(est_sat, p_gain=0.0001, d_gain=0.001, c_gain=0.001,
                          h_target=np.array([0.0, 0.0, 0.0]))

# With RW (QP allocation)
controller = MTQ_w_RW_QP(est_sat, p_gain=0.0001, d_gain=0.001, c_gain=0.001,
                          h_target=np.array([0.0, 0.0, 0.0]))
```

## TODO

- [ ] Run full MC experiments (100 trials, 500s each)
- [ ] Implement actual UKF/USQUE for Chapter 4
- [ ] Add Wisniewski_twist controller
- [ ] Add spinning solution with body-frame disturbance
- [ ] Validate against thesis expected results
