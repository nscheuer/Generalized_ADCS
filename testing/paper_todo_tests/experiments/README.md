# Paper Experiments - Complete Figure Generation

This directory contains scripts to generate ALL figures and data for 4 academic papers.

## Quick Start

```bash
# List all available figures
python generate_all_paper_figures.py --list

# Run quick test (10 trials, 200s) - ~5 min per paper
python generate_all_paper_figures.py --paper 3p1 --quick

# Run full experiments (100 trials, 1000s) - ~2-4 hours per paper
python generate_all_paper_figures.py --paper 3p1 --full

# Generate specific figure
python generate_all_paper_figures.py --figure architecture --quick

# Generate ALL figures for ALL papers
python generate_all_paper_figures.py --all --full --output-dir ./paper_figures
```

## Available Figures

### 3+1 Paper (SmallSat EU / JOSS)
| Figure ID | Description | Output Files |
|-----------|-------------|--------------|
| `architecture` | 3+0 vs 3+1 vs 3+3 Monte Carlo | `fig_3p1_error_trajectories`, `fig_3p1_cdf`, `fig_3p1_success_rates`, `fig_3p1_histogram`, `table_3p1_architecture.tex` |
| `torque_envelope` | Achievable torque sets | `fig_3p1_torque_envelope` |
| `momentum` | Momentum management comparison | `fig_3p1_momentum_management` |
| `degradation` | Wheel failure graceful degradation | `fig_3p1_graceful_degradation` |

### Generalized Control Paper (SmallSat US / JGCD)
| Figure ID | Description | Output Files |
|-----------|-------------|--------------|
| `lp_vs_qp` | LP vs QP allocation comparison | `fig_lp_vs_qp_trajectories`, `fig_lp_vs_qp_cdf` |
| `direction` | Direction preservation analysis | `fig_direction_preservation`, `fig_direction_error_histogram` |
| `versatility` | Same control law, different actuators | `fig_framework_versatility` |
| `controllability` | Controllability vs orbit inclination | `fig_controllability_vs_inclination` |

### Planner Paper (SmallSat USA / JGCD)
| Figure ID | Description | Output Files |
|-----------|-------------|--------------|
| `pd_baseline` | PD control baseline results | `fig_pd_baseline_trajectories` |
| `altro` | ALTRO vs PD comparison | `fig_altro_vs_pd` |
| `multi_target` | Multi-target sequence | `fig_multi_target_sequence` |

### Package Paper (SmallSat EU / JOSS)
| Figure ID | Description | Output Files |
|-----------|-------------|--------------|
| `controllers` | Lovera vs Wisniewski comparison | `fig_controller_comparison` |
| `quickstart` | 5-minute demo result | `fig_quickstart_demo` |

## Critical Parameters

The experiments use thesis-correct parameters:

```python
# Duration: 1000s for proper MTQ convergence (NOT 200s!)
duration_s = 1000

# Goal type: REDUCED-ATTITUDE (thesis configuration)
use_reduced_attitude = True

# Controller gains
mtq_p_gain = 0.001   # For MTQ-only controllers
mtq_d_gain = 0.005
rw_p_gain = 0.0001   # For LP/QP controllers with RW
rw_d_gain = 0.001
rw_c_gain = 0.001    # Momentum management gain
```

## Expected Results (from Thesis)

When running with `--full` (100 trials, 1000s):

| Config | Goal | Mean Error | <1° | <5° | <10° |
|--------|------|------------|-----|-----|------|
| 3+0 (MTQ) | Reduced | ~10-20° | 11% | ~30% | 73% |
| 3+1 (Hybrid) | Reduced | ~2-5° | 73% | ~90% | ~95% |
| 3+3 (Full RW) | Reduced | ~0.2-1° | ~95% | 100% | 100% |

**Note:** Quick mode (200s) will NOT show proper convergence for MTQ-only systems!

## Output Directory Structure

```
paper_figures/
├── 3p1/
│   ├── fig_3p1_error_trajectories.pdf
│   ├── fig_3p1_cdf.pdf
│   ├── fig_3p1_success_rates.pdf
│   ├── fig_3p1_histogram.pdf
│   ├── fig_3p1_torque_envelope.pdf
│   ├── fig_3p1_momentum_management.pdf
│   ├── fig_3p1_graceful_degradation.pdf
│   ├── table_3p1_architecture.tex
│   └── data_3p1_architecture.json
├── generalized/
│   ├── fig_lp_vs_qp_*.pdf
│   ├── fig_direction_preservation.pdf
│   ├── fig_framework_versatility.pdf
│   └── fig_controllability_vs_inclination.pdf
├── planner/
│   ├── fig_pd_baseline_*.pdf
│   ├── fig_altro_vs_pd.pdf
│   └── fig_multi_target_sequence.pdf
└── package/
    ├── fig_controller_comparison.pdf
    └── fig_quickstart_demo.pdf
```

## Estimated Runtimes

| Experiment | Quick (10 trials, 200s) | Full (100 trials, 1000s) |
|------------|-------------------------|--------------------------|
| Architecture (3p1) | ~4 min | ~2 hours |
| LP vs QP | ~2 min | ~1 hour |
| Controller Comparison | ~2 min | ~1 hour |
| Torque Envelope | ~1 min | ~5 min |
| Momentum Management | ~2 min | ~10 min |
| Graceful Degradation | ~1 min | ~5 min |
| Controllability | ~10 min | ~1 hour |
| ALTRO (if available) | ~10 min | ~4-8 hours |

**Total for all figures:** ~20 min (quick) / ~8-12 hours (full)

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `generate_all_paper_figures.py` | **Main script** - generates all data figures |
| `generate_placeholder_figures.py` | Manual/schematic placeholders |
| `run_all_paper_experiments.py` | Legacy experiment runner |
| `thesis_chapter7_planning.py` | ALTRO Monte Carlo experiments |

## Figures Still Needing Manual Creation

Some figures cannot be auto-generated:

1. **Actuator Configuration Schematic** - Hand-drawn or CAD diagram showing 3+0, 3+1, 3+3 actuator placement
2. **Bolt-On Framework Diagram** - Architecture block diagram
3. **Decision Flowchart** - When to use which architecture
4. **Basilisk Comparison** - Requires installing and benchmarking Basilisk
5. **Raspberry Pi Benchmarks** - Requires running on actual hardware

Use `generate_placeholder_figures.py` for these until manual versions are ready.

## Troubleshooting

### "Ephemeris date out of range"
The code uses J2000 dates around 0.24 (year ~2024). If you see this error, check that the ephemeris file covers your date range.

### "Cannot reshape array"
Controller h_target must be shape (3,). The code handles this automatically.

### Long runtime
Use `--quick` for testing. Only use `--full` when generating final figures.

### MTQ-only not converging
MTQ systems need ~500-1000s to converge with realistic gains. Don't expect good results with 200s duration.
