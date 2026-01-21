# ALTRO Tuning Sweep Notes

## Goal
Tune ALTRO planner for `debug_plan_and_track_bc2.py` to achieve:
- **Speed**: < 60s (ideally < 20s) for 500s trajectory (ALTRO only, not orbit prop)
- **Quality**:
  - Mostly smooth trajectories (some bang-bang OK, no huge oscillations)
  - Near-zero angular error at end
  - Near-zero angular velocity at end
  - Converge quickly then hold at target
  - Respect actuator constraints
  - Good utilization of both RW and 3 MTQs

## Tunable Parameters (from planner_settings.py and planner_subsettings.py)

### Cost Weights (CostWeights class)
- `angle` / `angle_N`: Attitude error cost (running / terminal)
- `ang_vel` / `ang_vel_N`: Angular velocity cost (running / terminal)
- `control_mult`: Multiplier for all actuator costs
- `ang_cost_func_type`: 0=(1-dot), 1=0.5*(1-dot)^2, 2=acos(dot), 3=0.5*acos^2
- `use_raw_control_cost`: True=penalize |u|, False=penalize |u-u_prev|
- `use_full_cost_hessian`: True=full Newton, False=Gauss-Newton (faster but may be less accurate)

### Actuator Weights (PlannerSettings)
- `mtq_control_weight`: Default 1e3
- `rw_control_weight`: Default 1e5
- `rw_AM_weight`: Penalize RW momentum buildup (default 1e4)
- `rw_stic_weight`: Penalize low RW speeds (default 1e0)

### Convergence Settings (ConvergenceConfig)
- `max_outer_iter`: Aug Lag iterations (default 30)
- `max_inner_iter`: iLQR iterations per outer (default 250)
- `grad_tol`: Gradient norm tolerance (default 1e-4)
- `ilqr_cost_tol`: Cost change tolerance (default 1e-2)
- `c_max`: Max constraint violation (default 0.0002)

### Regularization (RegularizationConfig)
- `reg_init`: Initial regularization (default 1e-2)
- `reg_min`: Minimum regularization (default 1e-8)
- `use_dynamics_hess`: 0=no, 1=yes (default 1)

### Augmented Lagrangian (AugLagConfig)
- `penalty_init`: Initial penalty (Pass1 default 1e-3, Pass2 default 1e4)
- `penalty_scale`: Penalty increase factor (default 10)
- `penalty_max`: Max penalty (default 1e16)

### Timing
- `dt_tp`: Coarse trajectory planner timestep (e.g., 30s)
- `dt_tvlqr`: Fine TVLQR timestep (e.g., 1s)

### Initialization (InitTrajConfig)
- `bdot_gain`: Gain for initial guess generation
- `hl_angle_limit`: High/low angle threshold
- `high_settings` / `low_settings`: (gyro, damp, vel, quat, rand, umax)

## Current Settings in debug_plan_and_track_bc2.py
```python
dt_planning = 30  # Coarse
dt = 1  # Fine

# Cost weights
planner_settings.cost_main.ang_vel = 1e4
planner_settings.cost_main.ang_vel_N = 1e8
planner_settings.cost_main.angle = 1e10
planner_settings.cost_main.angle_N = 1e15

# Convergence
planner_settings.pass1.convergence.max_outer_iter = 30
planner_settings.pass1.convergence.max_inner_iter = 150
planner_settings.pass2.convergence.max_outer_iter = 20
planner_settings.pass2.convergence.max_inner_iter = 60

# Tolerances (relaxed for speed)
planner_settings.pass1.convergence.grad_tol = 0.005
planner_settings.pass1.convergence.ilqr_cost_tol = 0.01

# Hessians OFF for speed
planner_settings.cost_main.use_full_cost_hessian = False
planner_settings.pass1.regularization.use_dynamics_hess = 0
```

## Build Issue
The C++ trajectory planner module isn't built:
```
ModuleNotFoundError: No module named 'trajectory_planner.build.tplaunch'
```

Need to build the C++ extension first.

## Next Steps
1. Build the C++ planner module
2. Run baseline to measure current performance
3. Create tuning sweep script to systematically test parameter combinations
4. Document best configurations

## Session Log
- **2026-01-20 18:47**: Started investigation
- Read debug_plan_and_track_bc2.py, planner_settings.py, planner_subsettings.py
- Identified all tunable parameters
- Found build issue with C++ module
- **2026-01-20 18:50**: User says ALTRO C++ is rebuilding now - waiting for completion
- Will prepare tuning sweep script while waiting
- **2026-01-20 18:55**: Created `altro_tuning_sweep.py` with:
  - Predefined configs: baseline, fast, quality, balanced
  - Systematic sweeps over: dt_tp, iterations, Hessians, costs, bdot_on, penalties
  - TrajectoryMetrics class with scoring function
  - JSON output for results
  - Quick mode (--quick) for fast testing

## Files Created
- `debug/debug_controllers/debug_plan_and_track/altro_tuning_sweep.py` - Main tuning sweep script

## Key Insights from Code Review
1. **Two-pass optimization**: Pass 1 explores, Pass 2 refines
2. **bdot_on modes**: 0=off, 1=on, 2=smart (adaptive)
3. **Hessian options**: 
   - `use_full_cost_hessian`: True=Newton, False=Gauss-Newton (faster)
   - `use_dynamics_hess`: Include dynamics second derivatives
4. **Cost function types** (`ang_cost_func_type`):
   - 0 = (1 - q·q_goal) - linear
   - 1 = 0.5*(1 - q·q_goal)² - quadratic
   - 2 = acos(|q·q_goal|) - geodesic angle [RECOMMENDED]
   - 3 = 0.5*acos² - quadratic geodesic

## BLOCKING ISSUE - C++ Build Error
**File**: `trajectory_planner/src/planner/TinyMPC.cpp` line 261

**Problem**: Code expects 5 return values but `dynamicsJacobians` returns 3:
```cpp
// Line 261 - BROKEN:
auto [xdot, dist, dxdot_dx, dxdot_du, dxdot_dtorq] = sat.dynamicsJacobians(...)

// But dynamicsJacobians returns tuple<mat, mat, mat> (3 elements, not 5)
```

**Fix needed**: Either:
1. Change line 261 to expect 3 values: `auto [dxdot, dxdot_dx, dxdot_du] = ...`
2. Or update `dynamicsJacobians` to return 5 values

**Other usages** (in PlannerUtil.cpp) correctly expect 3 values:
```cpp
tuple<mat, mat, mat> jacK1 = sat.dynamicsJacobians(xk, uk, dynamics_info_k);
```

**Status**: Another agent is fixing this. Waiting for build to complete.
