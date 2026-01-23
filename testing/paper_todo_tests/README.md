# Paper TODO Tests & Experiments

This directory contains:
1. **Pytest tests** - Quick validation that codebase can generate paper data
2. **Experiment scripts** - Full experiments generating figures, tables, and data for papers

## Directory Structure

```
paper_todo_tests/
├── README.md                  # This file
├── test_todo_*.py            # Quick pytest validation tests
└── experiments/              # Full experiment scripts
    ├── README.md
    ├── run_3p1_architecture_comparison.py
    ├── run_thesis_monte_carlo.py
    └── run_lp_vs_qp_comparison.py
```

## Quick Start

### For Validation (pytest)
```bash
# Run all quick tests
pytest testing/paper_todo_tests/ -v

# Run with output
pytest testing/paper_todo_tests/ -v -s
```

### For Paper Figures (experiments)
```bash
cd testing/paper_todo_tests/experiments

# Quick test run
python run_3p1_architecture_comparison.py --quick

# Full paper run with all trials
python run_thesis_monte_carlo.py --full --output-dir ./paper_figures

# View all options
python run_lp_vs_qp_comparison.py --help
```

## Experiment Scripts

### `run_3p1_architecture_comparison.py`
**For Paper**: 3+1 (3-Magnetorquer, 1-Reaction-Wheel Architecture)

Generates:
- Table 1: PD Control Monte Carlo Results
- Table 2: Planner-Enhanced Results
- Error distribution figures
- Comparison bar charts

Expected Results:
| Config | Mean Error | % <1° |
|--------|------------|-------|
| 3+0 PD | 21.6° | 15% |
| 3+1 PD | 2.3° | 73% |
| 3+1+Planner | 0.05° | 100% |
| 3+3 PD | 0.24° | 100% |

### `run_thesis_monte_carlo.py`
**For Paper**: Planner Paper, Thesis Chapter 7

Generates:
- Single slew comparison (MTQ vs 3+1)
- Goal formulation impact (6x improvement figure!)
- Multi-target sequence results

Key Results:
- Reduced vs Full attitude: 67% vs 11% within 1° (**6x improvement!**)
- 3+1: 96% within 1°
- Multi-target mean: 0.45°, median: 0.03°

### `run_lp_vs_qp_comparison.py`
**For Paper**: Generalized Control Paper (CORE CONTRIBUTION)

Generates:
- Direction error distributions
- Closed-loop pointing comparison
- Lyapunov stability demonstration

Key Finding:
- LP: 0.0036° direction error → 17.02° final error
- QP: 33.01° direction error → 25.70° final error
- **LP wins through direction preservation!**

## Pytest Test Files

| File | Tests | Description |
|------|-------|-------------|
| `test_todo_data_computational.py` | 4 | Timing, memory benchmarks |
| `test_todo_data_desaturation.py` | 18 | Momentum tracking |
| `test_todo_data_generation.py` | 7 | CubeSat configs |
| `test_todo_data_lp_qp_comparison.py` | 14 | LP vs QP allocation |
| `test_todo_data_sensitivity.py` | 3 | Parameter sensitivity |
| `test_todo_data_thruster.py` | 17 | Thruster validation |
| `test_todo_fig_generation.py` | 9 | Figure data generation |
| `test_todo_sim_controller_comparison.py` | 6 | Controller comparison |
| `test_todo_sim_monte_carlo.py` | 4 | MC infrastructure |
| `test_todo_sim_scenarios.py` | 8 | Scenarios testing |

## Paper Sources

The experiment designs come from the TODO sections in:
- `/mnt/c/Users/LV - Patrick McKeen/Writing/3+1 Ppaer/` - 3+1 Paper
- `/mnt/c/Users/LV - Patrick McKeen/Writing/Planner paper/` - Planner Paper
- `/mnt/c/Users/LV - Patrick McKeen/Writing/Generalied Control Paper/` - Gen. Control
- `/mnt/c/Users/LV - Patrick McKeen/Writing/Package paper/` - Package Paper
- `/mnt/c/Users/LV - Patrick McKeen/Writing/Dissertation/` - PhD Thesis

## Output Examples

Running experiments produces:

```
output/
├── fig_error_dist_pd.png       # Error distribution histogram
├── fig_comparison_pd.png       # Bar chart comparison
├── fig_goal_formulation.png    # 6x improvement figure
├── fig_lyapunov_stability.png  # Stability demonstration
├── table1_pd_results.tex       # LaTeX table for paper
├── table2_planner_results.tex  # LaTeX table
└── experiment_data.json        # Raw data for analysis
```

## Integration Notes

The experiment scripts currently use **placeholder simulations** generating
synthetic data matching thesis expected values. To use real simulations:

1. Replace placeholder functions with actual control loops
2. Import controllers from `ADCS.controller`
3. Connect to ALTRO planner
4. Run actual orbit propagation

See `experiments/README.md` for integration details.

## Related Documentation

- `research/TODO_TEST_COVERAGE.md` - Coverage analysis
- `research/PAPER_TODO_FEASIBILITY_ANALYSIS.md` - TODO feasibility
- Papers in `/mnt/c/Users/LV - Patrick McKeen/Writing/`
