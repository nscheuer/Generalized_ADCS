# Complete Figure and TODO Tracker

This document tracks ALL figures and TODOs from papers and thesis, mapping them to generation scripts.

## Papers Overview

### 1. 3+1 Paper (3-Magnetorquer, 1-Reaction-Wheel Architecture)
**Script:** `run_3p1_paper_experiments.py`

| TODO/Figure | Description | Status | Script Function |
|-------------|-------------|--------|-----------------|
| TODO-FIG-1 | Actuator config comparison (3+0, 3+1, 3+3) visual | ✅ | `generate_3p1_figures()` |
| TODO-FIG-2 | MC pointing error distributions (histogram/CDF) | ✅ | `generate_3p1_figures()` |
| TODO-FIG-3 | Time series 3+1 vs 3+0 vs 3+3 convergence | ✅ | `generate_3p1_figures()` |
| TODO-FIG-4 | Torque envelope comparison | 🔲 | TODO |
| TODO-FIG-5 | Momentum evolution continuous vs scheduled | 🔲 | TODO |
| TODO-FIG-6 | Decision flowchart for architecture | 🔲 | Manual |
| TODO-DATA-1 | Verify MC numbers match simulation | ✅ | Real sim |
| TODO-DATA-2 | Momentum management comparison | 🔲 | TODO |
| TODO-DATA-3 | Goal formulation comparison | ✅ | `run_goal_formulation_experiment.py` |
| TODO-DATA-4 | Actuator failure simulations | 🔲 | TODO |
| TODO-DATA-5 | Sensitivity analysis | 🔲 | TODO |
| TODO-DATA-6 | Spinning solution examples | ✅ | `generate_planning_figures()` |
| Table 1 | MC Results PD Control | ✅ | `generate_3p1_figures()` |
| Table 2 | MC Results with Planner | 🔲 | Needs ALTRO |
| Table 3 | Goal Formulation Impact | ✅ | `run_goal_formulation_experiment.py` |

### 2. Generalized Control Paper (LP vs QP Allocation)
**Script:** `run_lp_vs_qp_experiment.py`

| TODO/Figure | Description | Status | Script Function |
|-------------|-------------|--------|-----------------|
| TODO-FIG-1 | System block diagram | 🔲 | Manual |
| TODO-FIG-2 | Torque polytope evolution | 🔲 | TODO |
| TODO-FIG-3 | LP vs QP direction preservation | ✅ | `generate_lp_qp_figures()` |
| TODO-FIG-4 | Desaturation scheduling | 🔲 | TODO |
| TODO-FIG-5 | Pointing error time series | ✅ | `generate_lp_qp_figures()` |
| TODO-FIG-6 | MC pointing distributions | ✅ | `generate_lp_qp_figures()` |
| TODO-FIG-7 | Actuator failure response | 🔲 | TODO |
| TODO-FIG-8 | WCDTA sphere visualization | 🔲 | TODO |
| TODO-FIG-9 | Controllable subspace diagram | 🔲 | Manual |
| TODO-DAA-1 | Torque envelope one orbit | 🔲 | TODO |
| TODO-DAA-2 | WCDTA sphere configs | 🔲 | TODO |
| TODO-DATA-4 | LP vs QP comparison plots | ✅ | `generate_lp_qp_figures()` |
| TODO-DATA-5 | Desaturation performance | 🔲 | TODO |
| TODO-DATA-8 | Comparison table vs prior | ✅ | `generate_disturbance_figures()` |

### 3. Planner Paper (ALTRO Trajectory Planning)
**Script:** `generate_all_thesis_figures.py --chapter planning`

| TODO/Figure | Description | Status | Script Function |
|-------------|-------------|--------|-----------------|
| quaternion_set.png | Goal type visualization | 🔲 | Manual diagram |
| ALTRO_diagram.png | Algorithm structure | 🔲 | Manual diagram |
| ADCS_overview.png | Architecture diagram | 🔲 | Manual diagram |
| spinning_ang.png | Spinning solution error | ✅ | `generate_spinning_placeholder()` |
| spinning_av.png | Spinning angular velocity | ✅ | `generate_spinning_placeholder()` |
| spinning_cmd.png | Spinning commands | ✅ | `generate_spinning_placeholder()` |
| mtq_montecarlo.png | MTQ MC final errors | ✅ | `generate_planning_figures()` |
| 1W_montecarlo.png | 3+1 MC final errors | ✅ | `generate_planning_figures()` |
| *_montecarlo_traj.png | MC error trajectories | ✅ | `generate_planning_figures()` |
| TODO-DATA-1 | Hardware timing benchmarks | 🔲 | TODO |
| TODO-DATA-2 | Baseline controller comparison | ✅ | Wie/Lovera/Wisniewski |
| TODO-DATA-3 | Confidence intervals | 🔲 | TODO |
| TODO-DATA-4 | Convergence statistics | 🔲 | TODO |
| TODO-DATA-5 | Sensitivity analysis | 🔲 | TODO |

### 4. Package Paper (ADCS-Py Framework)
**Script:** N/A (mostly comparison tables)

| TODO/Figure | Description | Status | Script Function |
|-------------|-------------|--------|-----------------|
| TODO-DATA-1 | Framework benchmarks | 🔲 | TODO |
| TODO-DATA-2 | Comparative case studies | 🔲 | TODO |
| TODO-DATA-3 | Estimation accuracy | 🔲 | Estimation chapter |
| TODO-DATA-4 | Control law comparison | ✅ | Disturbance chapter |
| TODO-DATA-5 | Allocation comparison | ✅ | LP vs QP |
| TODO-JGCD-1 | Basilisk comparison | 🔲 | Manual |
| Table: Framework Comparison | ADCS-Py vs Basilisk | 🔲 | Manual |

---

## Thesis Chapters

### Chapter 4: Estimation (Cases A-G)
**Script:** `generate_all_thesis_figures.py --chapter estimation`

| Figure | Description | Status |
|--------|-------------|--------|
| log_angular_error_TRMM_initially_off.png | Case A angular error | 🔲 |
| usque_*_TRMM_initially_off_3sig.png | Case A USQUE bounds | 🔲 |
| mine_*_TRMM_initially_off_3sig.png | Case A DAF bounds | 🔲 |
| log_norm_av_TRMM_initially_off.png | Case A ang vel | 🔲 |
| log_angular_error_TRMM_initially_close.png | Case B angular error | 🔲 |
| log_angular_error_BC_initially_off.png | Case C angular error | 🔲 |
| mrp_BC_initially_off_3sig.png | Case C MRP bounds | 🔲 |
| axes_av_BC_initially_off_3sig.png | Case C ang vel bounds | 🔲 |
| (Cases D, E, F, G) | Various scenarios | 🔲 |

### Chapter 6: Disturbance Control
**Script:** `generate_all_thesis_figures.py --chapter disturbance`

| Figure | Description | Status |
|--------|-------------|--------|
| angular_error_wie.png | Wie comparison | ✅ |
| axes_av_wie.png | Wie ang vel | ✅ |
| ctrl_wie.png | Wie control effort | ✅ |
| angular_error_lovera.png | Lovera comparison | ✅ |
| ctrl_lovera.png | Lovera control | ✅ |
| log_angular_error_wisniewski.png | Wisniewski log error | ✅ |
| angular_error_wisniewski.png | Wisniewski error | ✅ |
| ctrl_wisniewski.png | Wisniewski control | ✅ |

### Chapter 7: Planning
**Script:** `generate_all_thesis_figures.py --chapter planning`

| Figure | Description | Status |
|--------|-------------|--------|
| anim_stack.png | Two-goal trajectory | 🔲 Needs ALTRO |
| anim_plots.png | Two-goal plots | 🔲 Needs ALTRO |
| spinning_ang.png | Spinning error | ✅ |
| spinning_av.png | Spinning ang vel | ✅ |
| spinning_cmd.png | Spinning commands | ✅ |
| simple_slew/mtq_montecarlo.png | MTQ MC histogram | ✅ |
| simple_slew/mtq_montecarlo_traj.png | MTQ MC trajectories | ✅ |
| simple_slew/1W_montecarlo.png | 3+1 MC histogram | ✅ |
| simple_slew/1W_montecarlo_traj.png | 3+1 MC trajectories | ✅ |
| simple_slew/1W_mom_montecarlo.png | 3+1 momentum | 🔲 |
| simple_slew/mtq_good_quaternion.png | MTQ good example | 🔲 |
| simple_slew/mtq_bad_quaternion.png | MTQ bad example | 🔲 |
| simple_slew/1W_good_quaternion.png | 3+1 good example | 🔲 |
| simple_slew/1W_bad_quaternion.png | 3+1 bad example | 🔲 |
| single_target_imaging/*.png | Reduced attitude MC | ✅ |
| multi_target_imaging/*.png | Multi-target MC | 🔲 |
| sequential/*.png | Sequential planning | 🔲 Needs ALTRO |

---

## Script Mapping

| Script | Generates | Status |
|--------|-----------|--------|
| `run_3p1_paper_experiments.py` | 3+1 Paper main figures | ✅ Working |
| `run_lp_vs_qp_experiment.py` | LP vs QP comparison | ✅ Working |
| `run_goal_formulation_experiment.py` | Goal formulation | ✅ Working |
| `generate_all_thesis_figures.py` | Master script | ✅ Working |

## Command Reference

```bash
# 3+1 Paper figures
python run_3p1_paper_experiments.py --full --output-dir ./output_3p1

# LP vs QP figures  
python run_lp_vs_qp_experiment.py --full --output-dir ./output_lp_qp

# Goal formulation
python run_goal_formulation_experiment.py --full --output-dir ./output_goal

# All thesis figures
python generate_all_thesis_figures.py --all --full --output-dir ./thesis_figures

# Specific chapters
python generate_all_thesis_figures.py --chapter planning --quick
python generate_all_thesis_figures.py --chapter disturbance --quick
python generate_all_thesis_figures.py --paper 3p1 --quick
python generate_all_thesis_figures.py --paper generalized --quick
```

## Notes

1. **ALTRO Integration**: Many planning figures require the ALTRO trajectory planner. Currently using placeholder data.

2. **Estimation Chapter**: Requires running USQUE vs Dynamics-Aware Filter comparisons with TRMM/CubeSat parameters.

3. **Manual Figures**: Block diagrams, flowcharts, and conceptual figures need manual creation (TikZ/Inkscape).

4. **Real Simulations**: All MC experiments now use real simulation code from ADCS package.
