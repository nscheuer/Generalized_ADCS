# Paper TODO Tests & Experiments

This directory contains:
1. **Pytest tests** (`test_todo_*.py`) - Quick validation tests
2. **Experiment scripts** (`experiments/`) - Full MC simulations generating publication figures

## Directory Structure

```
paper_todo_tests/
├── README.md                     # This file
├── test_todo_*.py               # Quick pytest validation tests
└── experiments/
    ├── README.md                # Experiment documentation
    ├── experiment_utils.py      # Shared utilities
    ├── run_3p1_paper_experiments.py    # 3+1 Paper experiments
    ├── run_lp_vs_qp_experiment.py      # LP vs QP comparison
    └── run_goal_formulation_experiment.py  # Goal formulation
```

## Quick Start

### Pytest Validation
```bash
cd /home/pmckeen/Generalized_ADCS

# Run all quick tests
pytest testing/paper_todo_tests/ -v

# Run specific test file
pytest testing/paper_todo_tests/test_todo_data_generation.py -v
```

### Figure Generation (Real Simulations!)
```bash
cd /home/pmckeen/Generalized_ADCS

# 3+1 Paper - Architecture Comparison (3+0 vs 3+1 vs 3+3)
python testing/paper_todo_tests/experiments/run_3p1_paper_experiments.py --quick
python testing/paper_todo_tests/experiments/run_3p1_paper_experiments.py --full --output-dir ./figures/3p1

# Generalized Control Paper - LP vs QP Torque Allocation
python testing/paper_todo_tests/experiments/run_lp_vs_qp_experiment.py --quick
python testing/paper_todo_tests/experiments/run_lp_vs_qp_experiment.py --full --output-dir ./figures/lp_qp

# Planner Paper - Goal Formulation Comparison
python testing/paper_todo_tests/experiments/run_goal_formulation_experiment.py --quick
```

## Experiment Scripts

All scripts use **real simulation code** from the ADCS package!

### `run_3p1_paper_experiments.py`
**Paper:** 3-Magnetorquer, 1-Reaction-Wheel Architecture

Compares:
- 3+0: 3 MTQs only (MTQ_Lovera controller)
- 3+1: 3 MTQs + 1 RW (MTQ_w_RW_LP controller)
- 3+3: 3 MTQs + 3 RWs (baseline)

**Outputs:**
- `fig_error_trajectories.{png,pdf}` - Time series with MC envelope
- `fig_error_histogram.{png,pdf}` - Final error distribution  
- `fig_success_rates.{png,pdf}` - Bar chart of success rates
- `fig_cdf.{png,pdf}` - Cumulative distribution
- `table1_results.tex` - LaTeX table

### `run_lp_vs_qp_experiment.py`
**Paper:** Generalized Attitude Control System

**CORE CONTRIBUTION:** LP-based torque allocation preserves direction.

**Key Thesis Results (Table 5.1):**
- LP: Direction error = 0.0036° (preserves direction)
- QP: Direction error = 33.01° (direction-wrong when infeasible)

**Outputs:**
- `fig_lp_vs_qp_trajectories.{png,pdf}`
- `fig_lp_vs_qp_boxplot.{png,pdf}`
- `fig_lp_vs_qp_cdf.{png,pdf}`
- `table_lp_vs_qp.tex`

### `run_goal_formulation_experiment.py`
**Paper:** Trajectory Planning for Magnetically Actuated Spacecraft

**KEY INSIGHT:** Reduced attitude (2-DOF) is much easier than full (3-DOF).

**Expected Results (with planner):**
- Reduced: 67% within 1°
- Full: 11% within 1°
- **6x improvement!**

**Note:** Full improvement requires trajectory planner (ALTRO), not just feedback control.

## Pytest Test Coverage

| File | Tests | Description |
|------|-------|-------------|
| `test_todo_data_generation.py` | 7 | CubeSat config validation |
| `test_todo_data_computational.py` | 4 | Timing benchmarks |
| `test_todo_data_desaturation.py` | 18 | Momentum management |
| `test_todo_data_lp_qp_comparison.py` | 14 | LP vs QP allocation |
| `test_todo_data_sensitivity.py` | 3 | Parameter sensitivity |
| `test_todo_data_thruster.py` | 17 | Thruster models |
| `test_todo_fig_generation.py` | 9 | Figure data structures |
| `test_todo_sim_controller_comparison.py` | 6 | Controller comparison |
| `test_todo_sim_monte_carlo.py` | 4 | MC infrastructure |
| `test_todo_sim_scenarios.py` | 8 | Mission scenarios |

## Paper Sources

These experiments implement TODOs from:
- `3+1 Ppaer/` - 3+1 Architecture Paper
- `Planner paper/` - Trajectory Planning Paper
- `Generalied Control Paper/` - Generalized Control Paper
- `Package paper/` - ADCS Package Paper
- `Dissertation/` - PhD Thesis

Located at: `/mnt/c/Users/LV - Patrick McKeen/Writing/`

## Command Line Options

All experiment scripts support:
- `--quick` - Fast validation (10 trials, shorter sim)
- `--full` - Publication quality (100 trials, full sim)
- `--output-dir DIR` - Output directory

## Output Format

Scripts generate:
- **PNG** (300 DPI) - For presentations
- **PDF** - For LaTeX papers
- **LaTeX tables** - Copy-paste into papers
- **JSON data** - Raw results for analysis

## Existing MC Infrastructure

For multi-core MC runs with progress dashboard, see:
- `papers/3MTQ+1RW/generate_bc2_lp.py` - LP controller
- `papers/3MTQ+1RW/generate_bc2_trajectory.py` - With planner
- Uses `ADCS.helpers.mc.monte_carlo_runner.MonteCarloRunner`
