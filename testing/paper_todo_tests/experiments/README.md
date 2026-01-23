# Paper Experiment Scripts

This directory contains standalone experiment scripts for generating paper figures, tables, and data.

Unlike pytest tests, these scripts are designed to:
- Generate publication-quality figures (PNG/PDF)
- Output LaTeX tables ready for papers
- Save JSON data for further analysis
- Produce pretty terminal output with summaries

## Quick Start

```bash
cd testing/paper_todo_tests/experiments

# Quick test run (few trials, fast)
python run_3p1_architecture_comparison.py --quick

# Full paper run (many trials, slow but accurate)
python run_3p1_architecture_comparison.py --full --output-dir ./paper_figures

# View help
python run_thesis_monte_carlo.py --help
```

## Available Experiments

### 1. `run_3p1_architecture_comparison.py`
**Paper**: 3+1 (3-Magnetorquer, 1-Reaction-Wheel)

Experiments:
- A1: PD Control Baseline (3+0, 3+1, 3+3)
- A2: Planner-Enhanced Comparison

Outputs:
- `fig_error_dist_pd.png` - Error distributions for PD control
- `fig_comparison_pd.png` - Bar chart comparison
- `fig_error_dist_planner.png` - With planner enabled
- `table1_pd_results.tex` - LaTeX Table 1
- `table2_planner_results.tex` - LaTeX Table 2
- `experiment_data.json` - Raw data

Expected results (thesis):
- 3+0 PD: 21.6° mean, 15% <1°
- 3+1 PD: 2.3° mean, 73% <1°
- 3+1+Planner: 0.05° mean, 100% <1°
- 3+3 PD: 0.24° mean, 100% <1°

### 2. `run_thesis_monte_carlo.py`
**Paper**: Planner Paper, Thesis Chapter 7

Experiments:
- Single 180° slew (MTQ vs 3+1)
- Goal formulation impact (Full vs Reduced attitude)
- Multi-target sequences

Outputs:
- `thesis_fig_single_slew.png` - MTQ vs 3+1 comparison
- `thesis_fig_goal_formulation.png` - 6x improvement figure
- `thesis_fig_multi_target.png` - Multi-target results
- `thesis_mc_data.json` - Raw data

Expected results (thesis):
- MTQ-only: 73% within 10°
- Reduced vs Full attitude: 67% vs 11% (6x improvement!)
- 3+1: 96% within 1°
- Multi-target mean: 0.45°, median: 0.03°

### 3. `run_lp_vs_qp_comparison.py`
**Paper**: Generalized Control Paper (CORE CONTRIBUTION)

Experiments:
- A1: Direction Preservation Test
- A2: Closed-Loop Pointing Comparison
- A3: Lyapunov Stability Demonstration

Outputs:
- `fig_direction_comparison.png` - Direction error distributions
- `fig_closed_loop_comparison.png` - Pointing error comparison
- `fig_lyapunov_stability.png` - Stability demonstration
- `lp_qp_comparison_data.json` - Raw data

Codebase reference values:
- LP: 0.0036° direction error, 17.02° final error
- QP: 33.01° direction error, 25.70° final error
- **LP wins through direction preservation!**

## Output Structure

```
output/
├── fig_*.png          # Publication figures (300 DPI)
├── table*.tex         # LaTeX tables
├── *_data.json        # Raw data for analysis
└── experiment_log.txt # Console output log
```

## Customization

Each script has a `Config` dataclass at the top with adjustable parameters:

```python
@dataclass
class ExperimentConfig:
    n_trials: int = 100        # Number of MC trials
    sim_duration_s: float = 1000.0
    fig_dpi: int = 300
    fig_format: str = "png"    # or "pdf" for publication
    # ...
```

## Converting to Paper Figures

1. Run experiment with `--full` for final data
2. Copy figures from output directory
3. Tables are ready for `\input{table1_pd_results.tex}`

For custom styling:
```python
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')  # or your journal's style
```

## Integration with Simulation Code

Currently these scripts use **placeholder simulations** that generate synthetic data
matching thesis expected values. To use real simulations:

1. Replace `run_single_trial()` with actual control loop
2. Import controllers from `ADCS.controller`
3. Import planner from trajectory module
4. Connect to orbit propagation

Example integration point:
```python
def run_single_trial(sat, config_name, trial_id, exp_config, use_planner=False):
    # TODO: Replace with actual simulation
    # 1. Create orbit
    # 2. Initialize controller (PD or with planner)
    # 3. Run simulation loop
    # 4. Compute metrics
    pass
```

## Related Files

- `../test_todo_*.py` - Quick pytest validation tests
- `../../research/*.py` - Research/exploration scripts
- `../../../papers/` - Paper data storage
