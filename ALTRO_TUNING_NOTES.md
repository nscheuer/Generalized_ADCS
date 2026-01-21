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

---

## Session 3 - 2026-01-20 Late Evening

### Bug Fix: debug_plan_and_track_bc2.py Not Working

**Symptoms**: 
- Script produced NaN/inf in ALTRO optimization
- State blew up to ~10^289 at timestep 1

**Root Causes Found & Fixed**:

1. **Orbit vectors in wrong units** (`test_plan_and_track_lqr` function):
   ```python
   # WRONG:
   R = 7000 * np.array([...])   # km, but should be meters
   V = np.array([8, 0, 0])      # 8 m/s, should be ~8 km/s
   
   # FIXED:
   R = 7000e3 * np.array([...])  # meters
   V = np.array([8000, 0, 0])    # m/s
   ```

2. **Old planner settings** - still had bdot_on=2 and many outdated settings

3. **Removed monkey-patch timing code** - simplified to use standard planner

### Current Working debug_plan_and_track_bc2.py Settings
```python
if __name__ == "__main__":
    plot_plan_and_track_lqr(
        verbose=0,
        tf=60,
        dt=1,
        dt_planning=30,
        real_orbit=False,  # use_J2=False for speed
        seed=42,
    )

# Planner settings (same as quick_planner_tests):
ps = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=dt_planning, dt_tvlqr=dt)
ps.cost_main.use_full_cost_hessian = True
ps.pass1.regularization.use_dynamics_hess = 1
ps.init_traj.bdot_gain = 500
ps.cost_main.angle = 100
ps.cost_main.angle_N = 5000
ps.pass1.aug_lag.penalty_init = 100
ps.pass1.convergence.max_outer_iter = 8
ps.pass1.convergence.max_inner_iter = 40
ps.pass2.convergence.max_outer_iter = 20
```

### Test Results
- **Planning**: 60s trajectory planned successfully
  - Final planned error: 3.36°
  - MTQ max: 0.177 A·m² (88% of 0.2 limit)
  
- **Simulation**: Tracking diverges
  - Final simulated error: 164° (planned vs actual mismatch)
  - This is a TVLQR tracking issue, not ALTRO planning issue

---

## Session 4 - 2026-01-20 Late Evening (Continued)

### Final Optimized Settings for 500s Trajectory

```python
ps = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=100, dt_tvlqr=1)
ps.cost_main.use_full_cost_hessian = True
ps.pass1.regularization.use_dynamics_hess = 1
ps.init_traj.bdot_gain = 500
ps.cost_main.angle = 100
ps.cost_main.angle_N = 50000  # KEY: Higher terminal cost for accuracy
ps.pass1.aug_lag.penalty_init = 100
ps.pass1.convergence.max_outer_iter = 8
ps.pass1.convergence.max_inner_iter = 40
ps.pass2.convergence.max_outer_iter = 5
ps.pass2.convergence.max_inner_iter = 15
```

**Result**: 29.2s solve time, 0.00° final error ✅

### Key Finding: angle_N Must Be High for Long Trajectories
| angle_N | Time | Error | Notes |
|---------|------|-------|-------|
| 10000 | 39.9s | 23.10° | Insufficient terminal penalty |
| 50000 | 29.2s | 0.00° | **Optimal** |

### Verbosity Levels Updated
- Level 0: Silent
- Level 1: Outer iterations + convergence summary  
- Level 2: cmax, grad, dLA, LA after each outer iter
- Level 3: z values, inner iteration details, line search
- Level 4: Debug matrices (removed full control printing)

### Next Steps
- TVLQR tracking now works - see findings below

---

## TVLQR Tracking - Root Cause Analysis (RESOLVED 2026-01-21)

### Original Symptom
- Planned trajectory: converges to 0° error at end
- Open-loop simulation (using u_ref from trajectory): diverges to ~1-2° error
- TVLQR feedback made tracking WORSE (oscillations)

### Root Causes Found

#### 1. Integration Method Mismatch (NOT A BUG - BY DESIGN)
- C++ planner uses RK4 integration
- Python simulation uses RK45 (higher accuracy) **intentionally** to test robustness
- With RK4 in Python (`sat.noiseless_rk4`): 0.0001° final error (exact match)
- With RK45 in Python (`solve_ivp`): ~0.6° final error (acceptable mismatch)

**This is expected behavior** - the simulation deliberately uses higher-accuracy 
integration to verify the trajectory works even with model differences.

#### 2. TVLQR Cost Settings Issue (FIXED)
The `cost_tvlqr` defaulted to `cost_main`, but TVLQR tracking needs different tuning:

- **ALTRO (cost_main)**: Low control cost optimizes trajectory shape
- **TVLQR (cost_tvlqr)**: Needs HIGHER `control_mult` for stable tracking gains

The TVLQR gains are computed as: `K = (R + B'SB)^(-1) * B'SA`
where `R = lkuu ∝ control_mult`. Higher R → smaller K → less aggressive feedback.

**With default settings**: ||K||_F ≈ 38, caused oscillations
**With tuned settings**: ||K||_F ≈ 0.8-15, stable tracking

### Fix Applied to `debug_plan_and_track_bc2.py`

```python
from ADCS.controller.helpers import CostWeights
planner_settings.cost_tvlqr = CostWeights(
    angle=1e4,          # State error cost
    angle_N=1e5,        # Terminal state error cost
    ang_vel=1e4,        # Angular velocity cost
    ang_vel_N=1e5,      # Terminal angular velocity cost
    control_mult=1e4,   # CRITICAL: Higher than cost_main to reduce gain aggressiveness
    ang_cost_func_type=0,
    use_raw_control_cost=True,
)
```

### Recommended TVLQR Tuning Guidelines

| control_mult | ||K||_F | Behavior |
|--------------|---------|----------|
| 1 (default from cost_main) | ~38 | Aggressive, oscillations |
| 1e4 | ~15 | Moderate feedback, stable |
| 1e5 | ~1.5 | Light feedback, smooth |
| 1e6 | ~0.001 | Essentially open-loop |

**Recommended**: `control_mult = 1e4` to `1e5` for balanced tracking.

### Verification Results

With RK45 simulation (higher accuracy than planner):
- Open-loop (u_ref only): ~0.6-1.7° final error ✓
- TVLQR (tuned settings): ~0.6-1.7° final error ✓
- Planned trajectory: 0.00° final error

The ~1° discrepancy is expected due to integration method differences and confirms
the trajectory is robust to small model variations.

---

## Session 5 - 2026-01-21 Morning

### NEW GOAL: Early Convergence for 500s Trajectories

**Requirements**:
1. Solve time < 30s (ALTRO only, not simulation)
2. Final error ≈ 0°
3. **NEW**: Converge EARLY (before t=250s) and STAY converged
   - Error should drop quickly toward 0°
   - Should NOT drift back up after convergence

### Problem Discovered: Trajectories Converge Only at End

With previous "fast" settings (ang_cost_func_type=2, low running cost):
```
Error at t=0s:   29.17°
Error at t=50s:  21.08°
Error at t=100s: 34.48°   ← Getting WORSE!
Error at t=250s: 81.54°   ← Much worse!
Error at t=298s: 88.12°   ← Peak error
Error at t=500s: 0.80°    ← Only converges at very end
```

The optimizer finds a "shortcut" - it's cheaper to drift away and correct at the end
rather than maintaining pointing throughout. This is because:
1. Running cost (angle) is low relative to terminal cost (angle_N)
2. Geodesic cost function (type 2) may have gradient issues for long horizons

---

## Complete Parameter Space Reference

### Cost Weights (CostWeights class)
| Parameter | Default | Range | Impact | Notes |
|-----------|---------|-------|--------|-------|
| `angle` | 1e3 | 1e2 - 1e8 | HIGH | Running cost on attitude error |
| `angle_N` | 1e4 | 1e3 - 1e9 | HIGH | Terminal cost on attitude error |
| `ang_vel` | 1e4 | 1e2 - 1e6 | MEDIUM | Running cost on angular velocity |
| `ang_vel_N` | 1e5 | 1e3 - 1e7 | MEDIUM | Terminal cost on angular velocity |
| `control_mult` | 1.0 | 0.1 - 1e8 | HIGH | Multiplies ALL actuator costs |
| `ang_cost_func_type` | 2 | 0-3 | **CRITICAL** | See below |
| `use_full_cost_hessian` | True | T/F | MEDIUM | True=faster convergence |
| `use_raw_control_cost` | True | T/F | LOW | True=|u|, False=|u-u_prev| |

**ang_cost_func_type options**:
- 0 = (1 - q·q_goal) - **LINEAR, constant gradient** ← Best for long trajectories
- 1 = 0.5*(1 - q·q_goal)² - Quadratic, weak gradient near goal
- 2 = acos(|q·q_goal|) - Geodesic angle (radians) ← Default, good for short
- 3 = 0.5*acos² - Quadratic geodesic

### Actuator Weights (PlannerSettings)
| Parameter | Default | Range | Impact | Notes |
|-----------|---------|-------|--------|-------|
| `mtq_control_weight` | 1e3 | 1e0 - 1e6 | MEDIUM | Higher = less MTQ usage |
| `rw_control_weight` | 1e5 | 1e0 - 1e8 | MEDIUM | Higher = less RW usage |
| `rw_AM_weight` | 1e4 | 1e0 - 1e6 | LOW | Penalize RW momentum buildup |
| `rw_stic_weight` | 1e0 | 1e-2 - 1e2 | LOW | Penalize low RW speeds |

### Timing Parameters
| Parameter | Default | Range | Impact | Notes |
|-----------|---------|-------|--------|-------|
| `dt_tp` | 30 | 10-100 | **CRITICAL** | Coarse planning timestep |
| `dt_tvlqr` | 1 | 0.5-5 | LOW | Fine TVLQR timestep |

**dt_tp guidelines**:
- dt_tp=10: Very fine, slow (~200s for 500s traj), best quality
- dt_tp=30: Good for <200s trajectories
- dt_tp=50: Good balance for 500s trajectories
- dt_tp=100: Fast (~30s), may lose mid-trajectory accuracy

### Initial Trajectory (InitTrajConfig)
| Parameter | Default | Range | Impact | Notes |
|-----------|---------|-------|--------|-------|
| `bdot_on` | 1 | 0/1/2 | MEDIUM | 0=off, 1=on, 2=smart |
| `bdot_gain` | 1000 | 100-5000 | LOW | Lower is often better |
| `high_settings` | (0,-2,0,-0.005,0.1,0.5) | - | LOW | (gyro,damp,vel,quat,rand,umax) |
| `low_settings` | (0,-1e-4,0,-1e-5,0.1,0.5) | - | LOW | For small angles |

### Pass 1 Convergence (ConvergenceConfig)
| Parameter | Default | Range | Impact | Notes |
|-----------|---------|-------|--------|-------|
| `max_outer_iter` | 30 | 5-50 | HIGH | Aug Lag iterations |
| `max_inner_iter` | 250 | 30-300 | MEDIUM | iLQR per outer |
| `grad_tol` | 1e-4 | 1e-6 - 1e-1 | LOW | Gradient norm tolerance |
| `ilqr_cost_tol` | 1e-2 | 1e-4 - 0.5 | LOW | Cost change tolerance |
| `c_max` | 2e-4 | 1e-6 - 1e-2 | LOW | Max constraint violation |

### Pass 1 Aug Lag (AugLagConfig)
| Parameter | Default | Range | Impact | Notes |
|-----------|---------|-------|--------|-------|
| `penalty_init` | 1e-3 | 1e-4 - 1e3 | MEDIUM | Initial penalty |
| `penalty_scale` | 10 | 2-100 | LOW | Penalty increase factor |
| `penalty_max` | 1e16 | - | - | Usually don't change |

### Pass 1 Regularization (RegularizationConfig)
| Parameter | Default | Range | Impact | Notes |
|-----------|---------|-------|--------|-------|
| `reg_init` | 1e-2 | 1e-4 - 1 | LOW | Initial regularization |
| `use_dynamics_hess` | 1 | 0/1 | MEDIUM | 1=faster |
| `use_constraint_hess` | 0 | 0/1 | LOW | 0=slightly faster |

### Pass 2 Settings
Same structure as Pass 1, but defaults differ:
- `penalty_init`: 1e4 (higher to enforce constraints)
- Typically fewer iterations needed

---

## Experimental Results: 500s Trajectory Tuning

### Baseline (Previous "Fast" Settings)
```python
ang_cost_func_type=2, angle=10000, angle_N=50000, dt_tp=100
```
- Time: ~31s ✓
- Final error: 0.80° ✓
- **BUT**: Error peaks at 88° mid-trajectory, only converges at end ✗

### Experiment 1: Higher Running Cost (type=2)
```python
angle=1e6, angle_N=1e7
```
- Time: 61.6s ✗ (too slow)
- Final error: 1.57°
- Still diverges mid-trajectory (max 99° at t=297s) ✗

### Experiment 2: bdot_on=0 + Equal Running/Terminal
```python
bdot_on=0, angle=1e5, angle_N=1e5
```
- Time: 95.8s ✗
- Final error: 102.6° ✗ (never converges!)
- Starts converging (8° at t=75s) then drifts away

### Experiment 3: ang_cost_func_type=0 (LINEAR)  ← **BREAKTHROUGH**
```python
ang_cost_func_type=0, angle=1e8, angle_N=1e9, dt_tp=50
```
- Time: 149.8s ✗ (too slow)
- Final error: 0.41° ✓
- **Trajectory shape is CORRECT**: monotonically decreases!
  - t=0: 29°, t=50: 12°, t=100: 3.8°, t=150: 3.3°, t=500: 0.4°
- First <5° at t=45s ✓
- First <1° at t=136s ✓

**KEY INSIGHT**: Linear cost function (type=0) gives constant gradient,
preventing optimizer from finding "lazy" trajectories that drift mid-way.

### Experiment 4: Linear + Speed Tuning (dt_tp=100)
```python
ang_cost_func_type=0, angle=1e8, angle_N=1e9, dt_tp=100
pass1: outer=8, inner=50
pass2: outer=3, inner=15
```
- Time: 31.7s ✓
- Final error: 24.5° ✗
- Converges to 5° at t=79s then drifts back up ✗
- dt_tp=100 too coarse for linear cost

### Experiment 5: Linear + dt_tp=50 + Tight Iters
```python
ang_cost_func_type=0, angle=1e6, angle_N=1e7, dt_tp=50
pass1: outer=6, inner=40
pass2: outer=3, inner=15
grad_tol=1e-2, ilqr_cost_tol=0.1
```
- Time: 25.3s ✓
- Final error: 78.7° ✗ (completely diverges)
- Too aggressive on tolerances/iterations

### Experiment 6: Linear + dt_tp=50 + Balanced
```python
ang_cost_func_type=0, angle=1e7, angle_N=1e8, dt_tp=50
pass1: outer=10, inner=60
pass2: outer=5, inner=20
```
- Time: 66.4s ✗
- Final error: 1.87° ✓
- Good trajectory shape, stays <5° after t=67s ✓
- Converges at outer=3 in pass1 → can reduce iterations

### Experiment 7: Linear + Optimized Iterations
```python
ang_cost_func_type=0, angle=1e7, angle_N=1e8, dt_tp=50
pass1: outer=5, inner=50
pass2: outer=4, inner=15
```
- Time: 44.3s (getting closer!)
- Final error: 4.17° ✓
- First <5° at t=74s ✓
- Some mid-trajectory oscillation (5.5° at t=300s)

---

## Summary: Parameter Impact Ranking

### CRITICAL (must get right)
1. **ang_cost_func_type**: 0 (linear) for long trajectories with early convergence
2. **dt_tp**: 50 balances speed/quality for 500s; 100 is too coarse for linear cost
3. **angle / angle_N ratio**: High running cost (1e7+) needed for early convergence

### HIGH IMPACT
4. **pass1.max_outer_iter**: 5-10 usually sufficient
5. **pass1.max_inner_iter**: 40-60 for 500s
6. **use_full_cost_hessian**: True = faster convergence
7. **use_dynamics_hess**: 1 = faster

### MEDIUM IMPACT
8. **pass2 iterations**: Can be low (outer=3-5, inner=15-25)
9. **penalty_init**: 100 faster than 1e-3 for pass1
10. **bdot_on**: 2 (smart) works well, 0 can be faster

### LOW IMPACT (usually leave at defaults)
11. **grad_tol, ilqr_cost_tol**: Loosening too much hurts quality
12. **ang_vel / ang_vel_N**: 1e4 / 1e5 works well
13. **control_mult**: Leave at 1.0 for pass1
14. **bdot_gain**: 500 slightly better than 1000

---

## Current Best Configuration (for early convergence goal)

```python
ps = PlannerSettings(est_sat=sat, bdot_on=2, dt_tp=50, dt_tvlqr=1)

# LINEAR cost function - critical for early convergence!
ps.cost_main.ang_cost_func_type = 0
ps.cost_main.angle = 1e7
ps.cost_main.angle_N = 1e8
ps.cost_main.ang_vel = 1e4
ps.cost_main.ang_vel_N = 1e5

# Hessians for speed
ps.cost_main.use_full_cost_hessian = True
ps.pass1.regularization.use_dynamics_hess = 1

# Iteration limits
ps.pass1.convergence.max_outer_iter = 8
ps.pass1.convergence.max_inner_iter = 50
ps.pass2.convergence.max_outer_iter = 5
ps.pass2.convergence.max_inner_iter = 20

# Other tuning
ps.init_traj.bdot_gain = 500
ps.pass1.aug_lag.penalty_init = 100
```

**Expected results**: ~45-60s solve time, <5° final error, converges by t=100s

---

## Session 5 Continued: Speed vs Quality Tradeoffs

### The Core Tradeoff
With linear cost (type=0), we get excellent trajectory shape but slower solve times:

| dt_tp | Solve Time | Final Error | Early Convergence | Notes |
|-------|------------|-------------|-------------------|-------|
| 50 | ~75s | 0.22° | ✓ (<5° at t=55s) | Best quality, too slow |
| 100 | ~32s | 24.5° | ✗ (drifts after 100s) | Fast but poor quality |

The problem: With linear cost + coarse dt_tp, the optimizer doesn't have enough
knot points to maintain the trajectory shape mid-way.

### Options to Reach <30s with Good Trajectory

1. **Accept higher mid-trajectory error** (current dt_tp=100 geodesic approach)
   - Converges only at end, but meets speed and final error goals
   
2. **Use dt_tp=50 with aggressive iteration reduction**
   - Risk: quality degrades if iterations too low
   - Needs careful pass1/pass2 balance

3. **Hybrid approach**: geodesic (type=2) for speed, but with higher running costs
   - Not yet tested exhaustively
   
4. **Accept ~45-60s solve time** for best trajectory shape
   - Still faster than previous >100s baseline

### Recommended Settings by Priority

**Priority: Speed (<30s) + Final Accuracy**
```python
# Uses geodesic cost - converges late but fast
ps.cost_main.ang_cost_func_type = 2  
ps.cost_main.angle = 10000
ps.cost_main.angle_N = 50000
ps.dt_tp = 100
# ... (see earlier "FAST" settings)
```
- Solve: ~30s, Final: <1°, BUT error peaks ~88° mid-trajectory

**Priority: Early Convergence + Quality (allows ~60s)**
```python
# Uses linear cost - converges early, stays converged
ps.cost_main.ang_cost_func_type = 0
ps.cost_main.angle = 1e7
ps.cost_main.angle_N = 1e8
ps.dt_tp = 50
ps.pass1.convergence.max_outer_iter = 8
ps.pass1.convergence.max_inner_iter = 50
ps.pass2.convergence.max_outer_iter = 5
ps.pass2.convergence.max_inner_iter = 20
```
- Solve: ~60-75s, Final: <1°, converges by t=60s, stays <5°

### Key Insight Summary

The `ang_cost_func_type` is the **single most important parameter** for trajectory shape:
- Type 0 (linear): Constant gradient → monotonic convergence, slower
- Type 2 (geodesic): Variable gradient → can find "lazy" shortcuts, faster

For applications requiring early convergence and stability, use type 0 or type 1 and accept longer solve times. For applications only caring about final state, use type 2 for speed.

---

## Session 5 Final: Complete Cost Function Comparison

### All Four ang_cost_func_type Options Tested

| Type | Formula | Solve Time | Final Error | First <5° | Stays Converged | Recommendation |
|------|---------|------------|-------------|-----------|-----------------|----------------|
| **0** | (1 - q·q_goal) | 60-100s | **0.22°** | t=55s ✓ | Yes (max 4.9°) ✓ | **Best for <0.5° accuracy** |
| **1** | 0.5*(1 - q·q_goal)² | 63-72s | 4.12° | t=77s ✓ | Yes (max 5.0°) ✓ | **Good speed/quality balance** |
| 2 | acos(\|q·q_goal\|) | 58-80s | 80°+ | Never | No | Not recommended (dt_tp=50) |
| 3 | 0.5*acos² | 54-62s | 60°+ | Never | No | Not recommended (dt_tp=50) |

*All tested with dt_tp=50, angle=1e7, angle_N=1e8*

**Key Findings**:

1. **Type 0 (linear) is best for high accuracy**
   - Constant gradient → monotonic error decrease
   - First <5° at t=55s, final error 0.22°
   - Required for <0.5° final error requirement
   - Solve time ~60-100s

2. **Type 1 (quadratic dot) is a good speed/quality balance**
   - Similar trajectory shape to type 0 (converges early, stays converged)
   - First <5° at t=77s, final error ~4°
   - Faster at 63-72s
   - Good choice when ~5° accuracy is acceptable

3. **Types 2 & 3 don't work for early convergence** with dt_tp=50
   - They find "lazy" trajectories that drift and never converge
   - Would need dt_tp=100+ which loses mid-trajectory resolution
   - Can still be used if only final state matters (see Option C)

### Configuration Options

#### Option A: High Accuracy (RECOMMENDED for <0.5° final error)

For 500s trajectory with **<0.5° final error** and early convergence:

```python
ps = PlannerSettings(est_sat=sat, bdot_on=2, dt_tp=50, dt_tvlqr=1)

# Type 0 (linear) - required for <0.5° final error
ps.cost_main.ang_cost_func_type = 0
ps.cost_main.angle = 1e7
ps.cost_main.angle_N = 1e8
ps.cost_main.ang_vel = 1e4
ps.cost_main.ang_vel_N = 1e5

# Hessians
ps.cost_main.use_full_cost_hessian = True
ps.pass1.regularization.use_dynamics_hess = 1

# Iterations
ps.pass1.convergence.max_outer_iter = 8
ps.pass1.convergence.max_inner_iter = 50
ps.pass2.convergence.max_outer_iter = 5
ps.pass2.convergence.max_inner_iter = 20

# Other
ps.init_traj.bdot_gain = 500
ps.pass1.aug_lag.penalty_init = 100
```

**Expected Results**:
- Solve time: ~60-100s (varies)
- Final error: ~0.2° ✓
- First <5°: t ≈ 55s ✓
- First <0.5°: t ≈ 79s ✓
- Stays <5° after convergence ✓

#### Option B: Faster Speed (if ~5° final error is acceptable)

For applications where speed matters more than final accuracy:

```python
ps = PlannerSettings(est_sat=sat, bdot_on=2, dt_tp=50, dt_tvlqr=1)

# Type 1 (quadratic dot) - good balance of speed and quality
ps.cost_main.ang_cost_func_type = 1
ps.cost_main.angle = 1e7
ps.cost_main.angle_N = 1e8
ps.cost_main.ang_vel = 1e4
ps.cost_main.ang_vel_N = 1e5

# Same other settings as Option A
ps.cost_main.use_full_cost_hessian = True
ps.pass1.regularization.use_dynamics_hess = 1
ps.pass1.convergence.max_outer_iter = 8
ps.pass1.convergence.max_inner_iter = 50
ps.pass2.convergence.max_outer_iter = 5
ps.pass2.convergence.max_inner_iter = 20
ps.init_traj.bdot_gain = 500
ps.pass1.aug_lag.penalty_init = 100
```

**Expected Results**:
- Solve time: ~63-72s (slightly faster)
- Final error: ~4° 
- First <5°: t ≈ 77s ✓
- Stays <5° after convergence ✓

**Use Case**: When you need faster planning and can tolerate ~4-5° final error,
or when TVLQR tracking will refine the trajectory further.

#### Option C: Maximum Speed (terminal accuracy only)

For applications only caring about final state (not trajectory shape):

```python
ps = PlannerSettings(est_sat=sat, bdot_on=2, dt_tp=100, dt_tvlqr=1)

# Type 2 (geodesic) with previous tuning
ps.cost_main.ang_cost_func_type = 2
ps.cost_main.angle = 10000
ps.cost_main.angle_N = 50000
# ... reduced iterations for speed
```

**Expected Results**:
- Solve time: ~30s
- Final error: <1°
- BUT: Error peaks ~88° mid-trajectory, only converges at end

**Use Case**: When trajectory shape doesn't matter, only final pointing.

### Why Types 2 & 3 Fail for Early Convergence

The geodesic angle cost (acos) has a gradient that depends on the current error:
- Near 0°: gradient is large (good for fine control)
- Near 90°: gradient is smaller
- Near 180°: gradient approaches 0 (problematic!)

With a long trajectory (500s) and coarse dt_tp, the optimizer can find paths where
it drifts to ~90° error where the cost gradient is weak, then relies on terminal
cost to snap back at the end. This is "locally optimal" but not desired behavior.

Linear and quadratic-dot costs have gradients that don't depend on the error magnitude,
so they push toward the goal at every timestep equally.
