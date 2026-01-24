# Paper Experiments

This directory contains experiment scripts for generating all figures and data for 4 academic papers.

## Quick Start

```bash
# List all experiments
python run_all_paper_experiments.py --list

# Run quick tests (10 trials, 200s)
python run_all_paper_experiments.py --paper 3p1 --quick

# Run full experiments (100 trials, 1000s)
python run_all_paper_experiments.py --paper 3p1 --full --output-dir ./paper_figures

# Generate placeholder figures for items needing ALTRO
python generate_placeholder_figures.py --all
```

## Papers and Key Experiments

### 1. 3+1 Paper (SmallSat EU / JOSS)
**Title:** "3-Magnetorquer, 1-Reaction-Wheel Architecture Demonstration"

| Experiment | Description | Script |
|------------|-------------|--------|
| A1: Architecture Comparison | 3+0 vs 3+1 vs 3+3 Monte Carlo | `run_all_paper_experiments.py --paper 3p1` |
| A2: Planner-Enhanced | With ALTRO trajectory planning | Requires `thesis_planning_figures.py` |
| B1: Momentum Management | Continuous vs scheduled desat | PLACEHOLDER |
| C3: Graceful Degradation | Wheel failure simulation | PLACEHOLDER |

**Key Claims to Verify:**
- 3+1 achieves 2.3° mean error (PD), 0.05° (planner)
- 73% within 1° (PD), 100% (planner)

### 2. Generalized Control Paper (SmallSat US / JGCD)
**Title:** "Generalized Attitude Control Allocation"

| Experiment | Description | Script |
|------------|-------------|--------|
| LP vs QP | Allocation comparison | `run_all_paper_experiments.py --paper generalized --experiment lp_vs_qp` |
| Framework Versatility | Same control, different actuators | `run_all_paper_experiments.py --paper generalized --experiment versatility` |
| Direction Preservation | LP: 0.004°, QP: 33° | `generate_placeholder_figures.py --paper generalized` |

**Key Claims to Verify:**
- LP preserves torque direction (0.004° error)
- QP has large direction error (~33°)
- Same control law works across configs

### 3. Planner Paper (SmallSat USA / JGCD)
**Title:** "Feasibility-Aware Attitude Trajectory Planner"

| Experiment | Description | Script |
|------------|-------------|--------|
| PD Baseline | Reference for planner comparison | `run_all_paper_experiments.py --paper planner` |
| ALTRO MC | Full ALTRO Monte Carlo | `thesis_planning_figures.py` |
| Spinning Solution | Emergent spin behavior | PLACEHOLDER |
| Multi-Target | Sequential goals | PLACEHOLDER |

**Key Claims to Verify (from thesis):**
- MTQ-only: 73% within 10° (planner), 15% (PD)
- 3+1: 96% within 1° (planner), 73% (PD)
- Reduced-attitude improves success 6x

### 4. Package Paper (SmallSat EU / JOSS)
**Title:** "Modular ADCS Python Framework"

| Experiment | Description | Script |
|------------|-------------|--------|
| Controller Comparison | Lovera vs Wisniewski | `run_all_paper_experiments.py --paper package` |
| Basilisk Comparison | Setup complexity, runtime | NEEDS MANUAL |
| 5-Minute Demo | Installation validation | NEEDS MANUAL |

## Critical Parameters (from Thesis)

### Correct Experiment Settings
```python
# CRITICAL: Use these settings to match thesis results!

# Duration: 1000s for proper MTQ convergence (NOT 200s!)
duration_s = 1000

# Goal type: REDUCED-ATTITUDE (thesis configuration)
use_reduced_attitude = True  # Align body axis with ECI vector
# NOT: Full-attitude (exact quaternion) - much harder!

# Controller gains
mtq_p_gain = 0.001
mtq_d_gain = 0.005
rw_p_gain = 0.0001
rw_d_gain = 0.001
```

### ALTRO Planner Settings (for planning experiments)
```python
# From ALTRO_TUNING_NOTES.md Session 5-6
planner_settings.cost_main.ang_cost_func_type = 0  # LINEAR for early convergence
planner_settings.cost_main.angle = 1e7
planner_settings.cost_main.angle_N = 1e8
planner_settings.dt_tp = 50
planner_settings.cost_tvlqr.control_mult = 1e4  # Prevents oscillations
```

## Data Discrepancy Warning

**CURRENT DATA vs THESIS CLAIMS:**

| Source | Duration | Trials | Result |
|--------|----------|--------|--------|
| experiment_outputs/3p1_paper/ | 200s | 10 | 63° for 3+1 (WRONG!) |
| papers/3MTQ+1RW/output_data/ | 500s | 10 | 30° for 3+1 |
| **Thesis claims** | 1000s | 100 | **0.45° mean, 96% <1°** |

**Root cause:** Goal type (reduced vs full attitude) and duration (200s vs 1000s).

**Fix:** Use `run_all_paper_experiments.py --full` with `use_reduced_attitude=True`.

## Output Directory Structure

```
paper_figures/
├── 3p1/
│   ├── fig_3p1_error_trajectories.pdf
│   ├── fig_3p1_cdf.pdf
│   ├── fig_3p1_success_rates.pdf
│   ├── table_3p1_mc.tex
│   └── data_3p1_architecture.json
├── generalized/
│   ├── fig_lp_qp_*.pdf
│   ├── fig_versatility_*.pdf
│   └── fig_direction_preservation.pdf
├── planner/
│   ├── fig_pd_baseline_*.pdf
│   ├── fig_planner_vs_pd_placeholder.pdf
│   └── fig_spinning_solution_placeholder.pdf
└── package/
    ├── fig_controller_comparison_*.pdf
    └── fig_basilisk_comparison_placeholder.pdf
```

## Scripts Reference

| Script | Purpose | Status |
|--------|---------|--------|
| `run_all_paper_experiments.py` | Main experiment runner | ✅ Complete |
| `generate_placeholder_figures.py` | Placeholders for complex figures | ✅ Complete |
| `run_3p1_paper_experiments.py` | Legacy 3+1 experiments | ✅ Complete |
| `run_lp_vs_qp_experiment.py` | Legacy LP/QP comparison | ✅ Complete |
| `thesis_chapter7_planning.py` | ALTRO planning experiments | ✅ Complete |
| `thesis_chapter6_disturbance.py` | Disturbance control | ✅ Complete |
| `thesis_chapter4_estimation.py` | Estimation experiments | ✅ Complete |

## Figure TODO Checklist

### 3+1 Paper
- [x] Error trajectories (3+0, 3+1, 3+3)
- [x] CDF of pointing errors
- [x] Success rate bar chart
- [x] Error histogram
- [ ] Actuator configuration schematic (PLACEHOLDER)
- [ ] Momentum management comparison (PLACEHOLDER)
- [ ] Graceful degradation (PLACEHOLDER)

### Generalized Control Paper
- [x] LP vs QP trajectories
- [x] LP vs QP CDF
- [x] Direction preservation (PLACEHOLDER)
- [x] Bolt-on framework diagram (PLACEHOLDER)
- [ ] Controllability vs inclination
- [ ] Torque polytope visualization

### Planner Paper
- [x] PD baseline results
- [x] Planner vs PD comparison (PLACEHOLDER)
- [x] Spinning solution (PLACEHOLDER)
- [x] Multi-target sequence (PLACEHOLDER)
- [ ] ALTRO timing benchmarks
- [ ] TVLQR tracking figure

### Package Paper
- [x] Controller comparison (Lovera vs Wisniewski)
- [x] Basilisk comparison table (PLACEHOLDER)
- [x] Quickstart demo (PLACEHOLDER)
- [ ] Framework architecture diagram
- [ ] Raspberry Pi benchmarks

## Running Full Campaign

To generate all figures for all papers:

```bash
# 1. Generate simulation-based figures (takes ~2-4 hours for full)
python run_all_paper_experiments.py --all --full --output-dir ./paper_figures

# 2. Generate placeholder figures for items needing ALTRO
python generate_placeholder_figures.py --all --output-dir ./paper_figures/placeholders

# 3. Run ALTRO experiments (if planner working)
python thesis_planning_figures.py --config mc_180deg_1rw --quick
```

## Estimated Runtimes

| Experiment | Quick (10 trials, 200s) | Full (100 trials, 1000s) |
|------------|-------------------------|--------------------------|
| 3+1 Architecture | ~5 min | ~2 hours |
| LP vs QP | ~3 min | ~1 hour |
| Controller Comparison | ~3 min | ~1 hour |
| ALTRO MC (per config) | ~10 min | ~4-8 hours |
