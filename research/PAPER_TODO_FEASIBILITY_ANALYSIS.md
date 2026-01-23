# Paper TODO Feasibility Analysis

## Overview

This document analyzes TODOs from papers in the Writing folder, identifying which can be achieved using existing code in the Generalized_ADCS repository or with reasonable adaptations.

**Analysis Date:** January 23, 2026
**Last Updated:** January 23, 2026

## Recent Additions (This Session)

### New Thruster Actuator Class
- **File:** `ADCS/satellite_hardware/actuators/thruster.py`
- **Tests:** `testing/test_actuators/test_actuator_thruster.py` (36 tests)
- Supports: cold gas, monopropellant, bipropellant, pulsed electric
- Features: propellant tracking, minimum impulse bit, full Jacobian/Hessian support
- References: Sutton & Biblarz, Wertz & Larson, Lemmer (CubeSat propulsion)

### New Test Suites
1. **LP vs QP Allocation Comparison** (`test_allocation_comparison.py`) - 13 tests
   - Direction preservation tests for LP
   - Magnitude minimization tests for QP
   - B-field sweep tests
   - Saturation behavior tests
   - Timing benchmarks

2. **Desaturation Tests** (`test_desaturation.py`) - 12 tests
   - Momentum tracking tests
   - Controller configuration tests
   - Saturation behavior tests
   - Statistical tests

**Total New Tests:** 61 tests, all passing

---

## Summary: High-Feasibility TODOs by Paper

### Quick Reference Table

| Paper | Ready Now | Easy to Adapt | Needs New Code | Not Feasible |
|-------|-----------|---------------|----------------|--------------|
| Generalized Control | 18 | 12 | 8 | 7 |
| Package Paper | 15 | 10 | 5 | 5 |
| Planner Paper | 12 | 8 | 3 | 2 |

---

## 1. Generalized Control Paper (Generalized_ACS_MASTER)

### ✅ READY NOW (Can run with existing code)

#### Data Generation
| TODO | Description | Code Location | How to Run |
|------|-------------|---------------|------------|
| **TODO-DATA-3** | Monte Carlo (N=1000) for pointing accuracy | `ADCS/helpers/mc/monte_carlo_runner.py` | Existing MC framework with config generator |
| **TODO-DATA-4** | LP vs QP comparison plots | `research/allocation_comparison.py`, `research/lp_vs_qp_deep_analysis.py` | Scripts already exist |
| **TODO-DATA-5** | Desaturation performance plots | `research/desaturation_tradeoff_study.py`, `research/continuous_desaturation.py` | Multiple desaturation scripts ready |

#### Simulations
| TODO | Description | Code Location | How to Run |
|------|-------------|---------------|------------|
| **TODO-SIM-3** | Pointing error time series | `testing/test_controllers/test_plan_and_track.py` | Existing simulation infrastructure |
| **TODO-SIM-4** | Monte Carlo statistics | `ADCS/helpers/mc/monte_carlo_runner.py` | Use existing runner with pointing metrics |
| **TODO-SIM-5** | Inertial hold results | `testing/test_controllers/` | Change goal to ECI fixed target |
| **TODO-SIM-6** | Tracking time-varying target | `ADCS/CONOPS/goals.py` supports this | Create orbit-linked goal |
| **TODO-SIM-8** | LP vs QP comparison | `research/lp_vs_qp_deep_analysis.py` | Already implemented |

#### Figures
| TODO | Description | Code Location | How to Run |
|------|-------------|---------------|------------|
| **TODO-FIG-5** | Pointing error time series | `testing/benchmarks/benchmark_planner.py` | Extract from sim logs |
| **TODO-FIG-6** | MC pointing error distributions | Use MC runner output | Histogram/CDF plotting |
| **TODO-DAA-1** | Torque envelope over orbit | `ADCS/controller/mtq_w_rw_LP.py` has `compute_available_torque_envelope()` | Call for orbit positions |
| **TODO-DAA-2** | WCDTA sphere visualization | `research/allocation_comparison.py` has geometry code | Adapt for 3D sphere plots |

#### Controllability Analysis
| TODO | Description | Code Location | How to Run |
|------|-------------|---------------|------------|
| **TODO-INTERPRET-1** | Torque polytope shapes comparison | `mtq_w_rw_LP.py` polytope methods | Compare 3RW orthogonal vs 4RW pyramid |
| **TODO-LP-1** | LP allocation geometry figure | `mtq_w_rw_LP.py` visualization | Already has plotting capability |

---

### 🔧 EASY TO ADAPT (Minor modifications needed)

#### Data Generation
| TODO | Description | What's Needed | Effort |
|------|-------------|---------------|--------|
| **TODO-DATA-1** | LTI/LTV rank tables for CMG/thruster | Add CMG/thruster actuator classes | 2-4 hours |
| **TODO-DATA-2** | Torque envelope over multiple orbits | Loop `compute_available_torque_envelope()` | 1 hour |
| **TODO-DATA-6** | Actuator failure simulations | Set actuator `max_torque=0` mid-sim | 2 hours |
| **TODO-DATA-8** | Comparison table vs prior methods | Run Lovera/Wisniewski controllers (exist!) | 3 hours |
| **TODO-DATA-9** | Sensitivity analysis (inertia, B-field) | Perturb parameters in MC config | 3 hours |

#### Simulations
| TODO | Description | What's Needed | Effort |
|------|-------------|---------------|--------|
| **TODO-SIM-7** | Failure response plots | Trigger actuator disable at t=T | 2 hours |
| **TODO-SIM-9** | Comparison with Lovera/Wisniewski | Controllers exist: `mtq_lovera.py`, `mtq_wisniewski.py` | 3 hours |

#### Other
| TODO | Description | What's Needed | Effort |
|------|-------------|---------------|--------|
| **TODO-COMP-1** | Disturbance compensation examples | Use `Plan_and_Track_LQR_Disturbed` | 2 hours |
| **TODO-DAA-3** | Numerical examples for CubeSat configs | Define realistic CubeSat params | 2 hours |
| **TODO-DAA-4** | WCDTA comparison table | Run allocation for configs | 3 hours |

---

### ⚠️ NEEDS NEW CODE (Significant development)

| TODO | Description | Gap | Effort Estimate |
|------|-------------|-----|-----------------|
| **TODO-DATA-7** | Timing on flight hardware (ARM Cortex-M4, LEON3) | Need cross-compilation, hardware | 1-2 weeks |
| **TODO-DISC-1** | Flight hardware benchmarks | Same as above | 1-2 weeks |
| **TODO-FIG-7** | Actuator failure response | Need failure injection framework | 4-8 hours |
| **TODO-FIG-2** | Torque polytope animation | Matplotlib 3D animation | 6 hours |
| **TODO-FIG-3** | LP vs QP 2D projection | Custom visualization | 4 hours |
| **TODO-FIG-4** | Desaturation scheduling | Need scheduling algorithm | 8 hours |
| **TODO-QP-3** | Computational complexity comparison | Benchmarking framework | 4 hours |
| **TODO-DESAT-1 to -6** | Full desaturation analysis | Partial code exists | 1-2 days |

---

### ❌ NOT FEASIBLE (Theory/writing/external)

| TODO | Description | Why Not Feasible |
|------|-------------|------------------|
| **TODO-PROOF-1 to -5** | Lyapunov stability proofs | Mathematical derivation, not code |
| **TODO-LP-2, LP-3** | Formal stability proofs | Mathematical analysis |
| **TODO-QP-1, QP-2** | QP stability analysis | Mathematical analysis |
| **TODO-WRITE-1 to -5** | Notation, acknowledgments, etc. | Writing tasks |
| **TODO-JGCD-8** | CMG singularity avoidance | No CMG class exists |

---

## 2. Package Paper (Generalized_ADCS_Python_MASTER)

### ✅ READY NOW

| TODO | Description | Code Location |
|------|-------------|---------------|
| **TODO-DATA-1** | Framework benchmarks | `testing/benchmarks/benchmark_planner.py` |
| **TODO-DATA-4** | Control law comparison | Run all controllers on same scenario |
| **TODO-DATA-5** | Allocation method comparisons | `research/allocation_comparison.py` |
| **TODO-DATA-8** | Unit test coverage | `pytest --cov` exists |
| **TODO-JGCD-4** | Computational complexity | Timing in benchmark scripts |
| **TODO-SMALLSAT-2** | Practitioner metrics | Adapt benchmark output |

### 🔧 EASY TO ADAPT

| TODO | Description | What's Needed | Effort |
|------|-------------|---------------|--------|
| **TODO-DATA-2** | Comparative case studies | Run predefined scenarios | 4 hours |
| **TODO-DATA-3** | Estimation accuracy | UKF tests exist in `test_estimators/` | 3 hours |
| **TODO-DATA-6** | Memory/CPU profiling | `memory_profiler` + `cProfile` | 2 hours |
| **TODO-JGCD-3** | Stability/convergence analysis | Collect convergence data from sims | 4 hours |
| **TODO-SMALLSAT-1** | 5-minute demo | Write example script | 2 hours |
| **TODO-SMALLSAT-3** | Failure modes documentation | Run edge cases, document | 4 hours |
| **TODO-JGCD-6** | Python vs C++ performance | Compare Python planner vs C++ ALTRO | 3 hours |

### ⚠️ NEEDS NEW CODE

| TODO | Description | Gap |
|------|-------------|-----|
| **TODO-DATA-7** | Hardware-in-the-loop | Need HIL setup |
| **TODO-DATA-9** | TRMM flight data | Need flight data access |
| **TODO-JGCD-1** | Formal Basilisk comparison | Need Basilisk installation |
| **TODO-FIGURE** | Architecture diagram | Drawing task |

---

## 3. Planner Paper (MTQ_Planner_MASTER)

### ✅ READY NOW

| TODO | Description | Code Location |
|------|-------------|---------------|
| **TODO-DATA-1** | Hardware timing benchmarks | `trajectory_planner/benchmark_altro.py` |
| **TODO-DATA-4** | Convergence statistics | Log iterations from planner |
| **TODO-DATA-7** | Computational scaling | Vary timestep/horizon in benchmarks |
| **TODO-DATA-8** | Failure mode characterization | Collect from MC runs |

### 🔧 EASY TO ADAPT

| TODO | Description | What's Needed | Effort |
|------|-------------|---------------|--------|
| **TODO-DATA-2** | Baseline controller comparisons | Run Quat PD, B-cross on same scenarios | 4 hours |
| **TODO-DATA-3** | Confidence intervals | Bootstrap from MC results | 2 hours |
| **TODO-DATA-5** | Sensitivity analysis sweeps | Perturb inertia, B-field, alignment | 6 hours |
| **TODO-DATA-6** | Operational scenario sims | Imaging pass, momentum management | 4 hours |
| **TODO-DATA-9** | SWaP-C metrics | Memory profiling + power estimation | 3 hours |

---

## Recommended Priority Order

### Phase 1: Quick Wins (1-2 days total)
1. **TODO-DATA-4** (LP vs QP comparison) - Scripts exist
2. **TODO-DATA-5** (Desaturation plots) - Scripts exist  
3. **TODO-SIM-9** (Lovera/Wisniewski comparison) - Controllers exist
4. **TODO-DAA-1** (Torque envelope plots) - Method exists
5. **TODO-DATA-1** (Planner timing) - Benchmark exists

### Phase 2: MC Campaigns (3-5 days)
1. **TODO-DATA-3** (MC N=1000 pointing accuracy)
2. **TODO-SIM-4** (MC statistics)
3. **TODO-FIG-6** (MC distributions)
4. **TODO-DATA-3** (Planner confidence intervals)

### Phase 3: Comparison Studies (1 week)
1. **TODO-DATA-8** (Prior methods comparison table)
2. **TODO-DATA-2** (Package comparative case studies)
3. **TODO-DATA-6** (Actuator failure sims)
4. **TODO-DATA-9** (Sensitivity analysis)

### Phase 4: Documentation & Figures (1 week)
1. All **TODO-FIG** items
2. **TODO-SMALLSAT** practitioner documentation
3. Architecture diagrams

---

## Code Locations Quick Reference

```
ADCS/
├── controller/
│   ├── mtq_w_rw_LP.py      # LP allocation, polytope methods
│   ├── mtq_w_rw_QP.py      # QP allocation
│   ├── mtq_lovera.py       # Lovera-Astolfi controller
│   ├── mtq_wisniewski.py   # Wisniewski controller
│   └── plan_and_track_*.py # Trajectory tracking controllers
├── helpers/
│   └── mc/monte_carlo_runner.py  # MC framework
└── satellite_hardware/
    └── actuators/          # RW, MTQ classes

research/
├── allocation_comparison.py       # LP/QP comparison framework
├── lp_vs_qp_deep_analysis.py     # Detailed LP/QP analysis
├── desaturation_tradeoff_study.py # Desaturation experiments
└── comprehensive_validation.py    # Validation framework

testing/
├── benchmarks/benchmark_planner.py  # Planner benchmarks
└── test_controllers/               # Controller tests & scenarios

trajectory_planner/
└── benchmark_altro.py              # ALTRO C++ benchmarks
```

---

## Next Steps

To begin generating paper data, I recommend starting with:

```bash
# 1. Run LP vs QP comparison
python research/lp_vs_qp_deep_analysis.py

# 2. Run desaturation study
python research/desaturation_tradeoff_study.py

# 3. Run planner benchmarks
python trajectory_planner/benchmark_altro.py

# 4. Run controller comparison tests
pytest testing/test_controllers/test_controller_mtq_w_rw_lp.py -v
```

Would you like me to create specific scripts for any of these TODOs?
