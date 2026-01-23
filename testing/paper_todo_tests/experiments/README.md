# Paper Experiment Scripts

Publication-quality figure generation for thesis and paper experiments.

## Quick Start

```bash
cd /home/pmckeen/Generalized_ADCS

# 3+1 Paper - Architecture Comparison
python testing/paper_todo_tests/experiments/run_3p1_paper_experiments.py --quick --output-dir ./output_3p1

# Generalized Control Paper - LP vs QP Torque Allocation  
python testing/paper_todo_tests/experiments/run_lp_vs_qp_experiment.py --quick --output-dir ./output_lp_qp

# Planner Paper - Goal Formulation Comparison
python testing/paper_todo_tests/experiments/run_goal_formulation_experiment.py --quick --output-dir ./output_goal
```

## Experiments

### 1. 3+1 Paper: `run_3p1_paper_experiments.py`

**Paper:** 3-Magnetorquer, 1-Reaction-Wheel Architecture Demonstration

Compares control architectures:
- **3+0**: 3 MTQs only (Lovera controller)
- **3+1**: 3 MTQs + 1 RW (LP controller) - **main contribution**
- **3+3**: 3 MTQs + 3 RWs (full authority baseline)

**Experiments:**
- A1: PD Control Baseline Comparison (implemented ✓)
- A2: Planner-Enhanced Comparison (TODO)
- A3: Goal Formulation Impact (TODO)
- B1-B2: Desaturation and Long-Duration (TODO)
- C1-C3: Mission-Specific Scenarios (TODO)

**Outputs:**
- `fig_error_trajectories.{png,pdf}` - Time series with MC envelope
- `fig_error_histogram.{png,pdf}` - Final error distribution
- `fig_success_rates.{png,pdf}` - Bar chart of success rates
- `fig_cdf.{png,pdf}` - Cumulative distribution function
- `table1_results.tex` - LaTeX table

### 2. Generalized Control Paper: `run_lp_vs_qp_experiment.py`

**Paper:** Generalized Attitude Control System

**CORE Contribution:** LP-based torque allocation preserves direction better than QP.

**Key Results (from thesis Table 5.1):**
- LP: Direction error = 0.0036° (preserves direction)
- QP: Direction error = 33.01° (direction-wrong when infeasible)

**Outputs:**
- `fig_lp_vs_qp_trajectories.{png,pdf}` - Comparative MC trajectories
- `fig_lp_vs_qp_boxplot.{png,pdf}` - Final error box comparison
- `fig_lp_vs_qp_cdf.{png,pdf}` - CDF comparison
- `table_lp_vs_qp.tex` - LaTeX comparison table

### 3. Planner Paper: `run_goal_formulation_experiment.py`

**Paper:** Trajectory Planning for Magnetically Actuated Spacecraft

**KEY INSIGHT:** For imaging missions, you only need to point a camera at a target 
(reduced attitude / 2-DOF), not achieve exact orientation (full attitude / 3-DOF).

**Expected Results (from thesis Table 3.2):**
- Reduced Attitude: 67% within 1°
- Full Attitude: 11% within 1°
- **6x improvement!**

**Note:** This comparison requires the trajectory planner (ALTRO) to see the 
full improvement. With feedback control only, both modes are slow to converge.

**Outputs:**
- `fig_goal_formulation_trajectories.{png,pdf}` - Error comparison
- `fig_goal_formulation_success.{png,pdf}` - Success rate bar chart
- `fig_goal_formulation_cdf.{png,pdf}` - CDF comparison
- `table_goal_formulation.tex` - LaTeX table

### 4. Allocation Comparison: `research/allocation_comparison.py`

Direct torque allocation comparison (single-step, not closed-loop).
Best demonstrates LP vs QP direction preservation.

```bash
python research/allocation_comparison.py
```

## Command Line Options

All scripts support:
- `--quick`: Fast validation (10 trials, shorter sim time)
- `--full`: Publication quality (100 trials, full sim time)
- `--output-dir DIR`: Specify output directory

## Figure Style

All figures use:
- Serif fonts (Computer Modern when TeX available)
- 300 DPI for publication
- Both PNG and PDF formats
- Colorblind-friendly palette
- Clean axes (no top/right spines)

## Expected Results

### 3+1 Paper (Table 2 Expected Values)
| Config | Mean Error | <1° | <5° |
|--------|-----------|-----|-----|
| 3+0    | ~21.6°    | ~15%| -   |
| 3+1 PD | ~2.3°     | ~73%| -   |
| 3+1+Plan| ~0.05°   | ~100%| -  |
| 3+3    | ~0.24°    | ~100%| -  |

**Note:** Results depend on simulation duration. Use `--full` for accurate comparison.

### LP vs QP (Direction Preservation)
- LP direction error: ~0.004° (essentially zero)
- QP direction error: ~31-33° (large when infeasible)

## Adding New Experiments

1. Create new script in this directory
2. Follow the pattern in `run_3p1_paper_experiments.py`:
   - Use `ExpConfig` for trial/duration settings
   - Use `run_single_trial()` for simulation
   - Use `run_monte_carlo()` for MC campaigns
   - Generate figures with matplotlib
   - Save LaTeX tables and JSON data

## Dependencies

- numpy, scipy, matplotlib
- tqdm (progress bars)
- ADCS package (controllers, satellites, orbits)
