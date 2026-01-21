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

**Status**: FIXED by previous agent.

## Session 2 - 2026-01-20 19:10

### Issue Fixed: Mat::cols() out of bounds error
When running quick_planner_tests.py, tests were failing with:
```
Mat::cols(): indices out of bounds or incorrectly used
colMiss: 30
```

**Root Cause**: In `OldPlanner.cpp` line 390, the code assumed `Uset.n_cols >= 3`:
```cpp
mat UsetLong = join_rows(repelem(Uset.cols(0,Uset.n_cols-3),1,...
```
When trajectory duration was short (30s with dt_tp=30), `Uset` had only 2 columns, causing `Uset.n_cols-3 = -1`.

**Fix Applied** (OldPlanner.cpp ~line 390):
```cpp
// Handle edge case where Uset has fewer than 3 columns
mat UsetLong;
if(Uset.n_cols >= 3) {
  // Normal case
  UsetLong = join_rows(repelem(Uset.cols(0,Uset.n_cols-3),1,int(dt_prev/dt_tvlqr)),...);
} else if(Uset.n_cols == 2) {
  // Only 2 columns: replicate first column
  int nReps = max(1, int(Rset_tvlqr.n_cols) - 1);
  UsetLong = join_rows(repelem(Uset.col(0), 1, nReps), Uset.tail_cols(1));
} else {
  // Only 1 column
  int nReps = max(1, int(Rset_tvlqr.n_cols));
  UsetLong = repelem(Uset.col(0), 1, nReps);
}
```

**Also**: Removed `-flto` from CMakeLists.txt due to linker errors (`multiple prevailing defs for 'solve'`).

**Test Results**: All 5 quick_planner_tests pass now:
- Basic ALTRO: PASS (0.44s)
- High Angular Velocity: PASS (3.11s)  
- 90 Degree Slew: PASS (9.11s)
- Zero Initial Omega: PASS (2.06s)
- Trajectory Shape: PASS (0.38s)

### Systematic Timing Results 

| Duration | dt_tp=30 Time | dt_tp=50 Time | Final Error |
|----------|---------------|---------------|-------------|
| 60s | 3.33s | 3.16s | 1.62° |
| 120s | 9.93s | 12.37s | 3.66° |
| 180s | 11.93s | 4.64s | 6.10° |
| 240s | 32.01s | 13.23s | 16.38° |
| 300s | >180s (timeout) | - | - |
| 500s | >180s | 14-15s | 64.5° |

**Key Findings**:
1. **Solve time scales non-linearly** - 60s=3s, 120s=10s, 240s=32s, 300s=timeout
2. **Error increases with duration** - planner reaches goal early but drifts away
   - Analysis of 500s trajectory shows error=0.28° at t=50s, then drifts to 64.5° by t=500s
3. **dt_tp=50 faster for long trajectories** but doesn't improve convergence
4. **The planner optimizes initial acquisition but not maintenance**

**Root Cause Hypothesis**: The trajectory planner finds a path to reach the goal 
but doesn't maintain control effort to *stay* at the goal. The running cost on 
attitude error may be too low relative to control cost, causing it to "coast" 
after initial maneuver.

### ROOT CAUSE FOUND: MTQ Control Cost Bug

**The optimizer zeroes MTQ commands** because the cost function makes RW cheaper!

Cost calculation: `0.5 * cmd² * weight * control_mult`
- MTQ: 0.5 * 0.1² * 1000 = **5** (for 0.1 Am² dipole)
- RW: 0.5 * 0.001² * 100000 = **0.00005** (for 0.001 Nm torque)

**RW is 100,000x cheaper than MTQ in the cost function!**

The problem: MTQ produces torque via `B × m` (cross product of B-field and dipole).
A 0.1 Am² dipole in a 50 μT field produces ~5 μNm torque.
But RW can directly command torque, so 0.001 Nm = 1000 μNm.

The cost weights aren't normalized by torque authority, making MTQs prohibitively expensive.

**Actual Root Cause Found**: IGRF B-field calculation bug!

The `get_b_eci()` and `get_b_eci_orbit()` functions passed radius in **meters** 
to `ppigrf.igrf_gc()` which expects **kilometers**. Result: B-field ≈ 0 instead of ~35 μT.

**Fix Applied**:
- `ADCS/orbits/orbital_state.py`: `r_m / 1000.0` in `get_b_eci()`  
- `ADCS/orbits/orbit.py`: `geos[:,0]/1000.0` in `get_b_eci_orbit()`

### Results After Fix

With correct B-field (~35 μT), the planner now uses MTQs properly:
- **60s trajectory**: Time=17.8s, MTQ max=0.158 Am² (79% of 0.2), final error=0.00°
- **120s trajectory**: Time=33.7s, MTQ max=0.143 Am², final error=0.00°

The optimizer converges well (grad=1.4e-9) and trajectory reaches goal!

### Validation
All quick_planner_tests pass (5/5):
- Basic ALTRO: 2.1s
- High Angular Velocity: 4.3s
- 90 Degree Slew: 34.0s
- Zero Initial Omega: 5.3s
- Trajectory Shape: 2.8s

### Summary of Bug Fixes This Session

1. **OldPlanner.cpp Mat::cols() bounds error** (line 390)
   - Fixed edge case when Uset has < 3 columns for short trajectories

2. **CMakeLists.txt LTO linker error**
   - Removed `-flto` flag causing "multiple prevailing defs for 'solve'"

3. **CRITICAL: IGRF radius units bug**
   - `get_b_eci()` and `get_b_eci_orbit()` passed meters instead of km
   - Result: B-field ≈ 0, MTQs had no torque authority, optimizer zeroed them
   - Fix: divide radius by 1000 before calling ppigrf.igrf_gc()

### Parameter Impact Summary (60s trajectory tests)

#### Hessian Settings
| Parameter | Best Value | Impact |
|-----------|------------|--------|
| `use_full_cost_hessian` | True | **15.8s vs 32.2s** - 2x faster! |
| `use_dynamics_hess` | 1 | **14.7s vs 18.8s** - ~25% faster |
| `use_constraint_hess` | 0 | 14.7s vs 16.5s - slightly faster without |

**Best hessian combo**: cost=1, dyn=1, con=0 → 14.7s

#### Initial Trajectory Settings
| Parameter | Best Value | Impact |
|-----------|------------|--------|
| `bdot_on` | 0 | 16.5s (vs 19.1s for bdot_on=1) |
| `bdot_gain` | 500 | **12.7s** (vs 16.9s default=1000, 29.5s for 5000) |

**Key insight**: Lower bdot_gain helps - less aggressive initial guess converges faster.

#### Cost Weights (BIGGEST IMPACT!)
| angle | angle_N (ratio) | Time | Error | Notes |
|-------|-----------------|------|-------|-------|
| 1e3 (default) | 1e4 (10x) | 33s | 0.00° | Baseline |
| 100 | 1000 (10x) | 2.2s | 0.90° | Fast but inaccurate |
| 100 | 2000 (20x) | 2.0s | 0.29° | Good tradeoff |
| 100 | 5000 (50x) | 7.1s | 0.00° | **Best balance** |
| 100 | 10000 (100x) | 10.8s | 0.00° | Slower |

**Key insight**: Low running cost (angle=100) + high terminal cost (50-100x ratio) is fastest while maintaining accuracy. The optimizer spends less effort on intermediate states.

#### Pass2 Settings
| Setting | Value | Time | Notes |
|---------|-------|------|-------|
| pass2 outer=5 | | 12.1s | Slightly faster |
| pass2 outer=20 | | 7.3s | **Much faster!** |
| pass2 disabled | | TBD | May skip refinement |

**Key insight**: More pass2 outer iterations can be FASTER (counterintuitive) - likely converges in fewer total iterations.

### Current Best Settings
```python
ps = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=30, dt_tvlqr=1)
ps.cost_main.use_full_cost_hessian = True
ps.pass1.regularization.use_dynamics_hess = 1
ps.pass1.regularization.use_constraint_hess = 0
ps.init_traj.bdot_gain = 500
ps.cost_main.angle = 100
ps.cost_main.angle_N = 5000
ps.pass2.convergence.max_outer_iter = 20
```

### Scaling Results (tuned settings)
| Duration | Solve Time | Error | Notes |
|----------|------------|-------|-------|
| 60s | 11.9s | 0.00° | Similar to default |
| 90s | 21.6s | 0.00° | |
| 120s | 29.5s | 0.00° | |
| 180s | >60s | - | Needs larger dt_tp |

### Key Findings
1. **bdot_on=0 is fastest** - 15.7s vs 58.2s for bdot_on=1 (on 60s trajectory)
2. **Low running cost + high terminal cost** - angle=100, angle_N=5000 converges faster
3. **Full hessians help** - use_full_cost_hessian=True, use_dynamics_hess=1
4. **bdot_gain=500** - lower than default (1000) helps initial guess

### Additional Findings

#### Tolerances
| Setting | Impact |
|---------|--------|
| `grad_tol=1e-2, cost_tol=0.5` | Looser tolerances help ~20% |

#### Penalty Init
| Setting | Impact |
|---------|--------|
| `pass1.aug_lag.penalty_init=100` | Faster constraint enforcement |

#### Pass2 Settings (BIG IMPACT)
| Setting | Time (60s) | Notes |
|---------|------------|-------|
| default (o=20, i=60) | 12.6s | |
| o=5, i=30 | **7.6s** | 40% faster! |

### Current FAST Settings
```python
ps = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=30, dt_tvlqr=1)
ps.cost_main.use_full_cost_hessian = True
ps.pass1.regularization.use_dynamics_hess = 1
ps.init_traj.bdot_gain = 500
ps.cost_main.angle = 100
ps.cost_main.angle_N = 5000
ps.pass1.aug_lag.penalty_init = 100
ps.pass2.convergence.max_outer_iter = 5
ps.pass2.convergence.max_inner_iter = 30
```

### Scaling Results (FAST settings)
| Duration | dt_tp | Solve Time | Error |
|----------|-------|------------|-------|
| 60s | 30 | 10.4s | 0.00° |
| 90s | 30 | 20.1s | 0.00° |
| 120s | 30 | 21.9s | 0.00° |
| 180s | 30 | 35.6s | 0.00° |
| 240s | 40 | 38.7s | 0.00° |

### 500s ACHIEVED: 51.5s with 0.08° error!

Settings for 500s trajectory:
```python
ps = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=100, dt_tvlqr=1)
ps.cost_main.use_full_cost_hessian = True
ps.pass1.regularization.use_dynamics_hess = 1
ps.init_traj.bdot_gain = 500
ps.cost_main.angle = 100
ps.cost_main.angle_N = 10000  # Higher terminal for 500s
ps.pass1.aug_lag.penalty_init = 100
ps.pass1.convergence.max_outer_iter = 8
ps.pass1.convergence.max_inner_iter = 40
ps.pass2.convergence.max_outer_iter = 3
ps.pass2.convergence.max_inner_iter = 15
```

### Final Performance Summary
| Duration | dt_tp | Solve Time | Error |
|----------|-------|------------|-------|
| 60s | 30 | 10.4s | 0.00° |
| 90s | 30 | 20.1s | 0.00° |
| 120s | 30 | 21.9s | 0.00° |
| 180s | 30 | 35.6s | 0.00° |
| 240s | 40 | 38.7s | 0.00° |
| **500s** | **100** | **51.5s** | **0.08°** |

### Key Tuning Insights
1. **Hessians matter**: use_full_cost_hessian=True, use_dynamics_hess=1 (25% faster)
2. **Low running + high terminal cost**: angle=100, angle_N=5000-10000 
3. **Higher penalty_init**: 100 instead of 0.1 (faster constraint enforcement)
4. **Reduce pass2 iterations**: outer=3-5, inner=15-30 (40% faster)
5. **Scale dt_tp with duration**: 30 for <200s, 40 for 240s, 100 for 500s
6. **bdot_on=0**: Skip bdot initial guess generation
