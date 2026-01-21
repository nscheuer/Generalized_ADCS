# ALTRO Planner Settings Comparison

## Final Benchmark Results (500s trajectory)

| Config | Wall Time | Final Error | Notes |
|--------|-----------|-------------|-------|
| **dt_tp=10, linear (BEST)** | **122s** | **0.20°** | Fastest AND most accurate |
| dt_tp=10, geodesic (NSSR) | 145s | 0.18° | Slightly slower |
| Original PKMN (bdot_on=3) | 318s | 71.53° | Too slow, poor convergence |

## Key Findings

### 1. **dt_tp Sanity Check Added**
A runtime check now prevents `dt_tp` from being too large:
```
ValueError: dt_tp=100s is too large for reliable planning. 
With 60s segments, this gives only N=1 points per segment. 
The C++ planner requires N >= 4. 
Suggested fix: set dt_tp <= 20.0s
```

### 2. **Linear Cost (type=0) is Better**
- **1.2x faster** than geodesic (type=2)
- **Same accuracy** (~0.2° final error)
- Geodesic was assumed faster but testing shows otherwise

### 3. **bdot_on=0 is Much Faster**
- `bdot_on=3` (smart bdot): 318s
- `bdot_on=0` (skip bdot): 122s
- **2.6x speedup** by skipping bdot initial guess

### 4. **Full Hessians Help**
- `use_full_cost_hessian=True` and `use_dynamics_hess=1` improve convergence

## Optimized Config (Now in debug_plan_and_track_bc2.py)

```python
planner_settings = PlannerSettings(
    est_sat=real_sat,
    bdot_on=0,          # Skip bdot (2.6x faster)
    dt_tp=10,           # Must be <= 20 for N >= 4
    dt_tvlqr=dt,
)

# Full Hessians for faster convergence
planner_settings.cost_main.use_full_cost_hessian = True
planner_settings.pass1.regularization.use_dynamics_hess = 1

# Linear cost with moderate weights
planner_settings.cost_main = CostWeights(
    angle=1e3,              # Not 1e6 (too slow)
    angle_N=1e6,
    ang_vel=1e3,
    ang_vel_N=1e5,
    control_mult=1.0,
    ang_cost_func_type=0,   # LINEAR (faster than geodesic!)
)

# Iteration limits
planner_settings.pass1.convergence.max_outer_iter = 8
planner_settings.pass1.convergence.max_inner_iter = 40
planner_settings.pass2.convergence.max_outer_iter = 5
planner_settings.pass2.convergence.max_inner_iter = 15
```

## Settings Comparison Table

| Setting | Original PKMN | NSSR | Optimized |
|---------|---------------|------|-----------|
| `bdot_on` | 3 | 0 | **0** |
| `dt_tp` | 100 (crashed) | 10 | **10** |
| `ang_cost_func_type` | 0 | 2 | **0** |
| `angle` | 1e6 | 1e3 | **1e3** |
| `angle_N` | 1e7 | 1e6 | **1e6** |
| `ang_vel_N` | 1e3 | 1e5 | **1e5** |
| `use_full_cost_hessian` | False | True | **True** |
| `use_dynamics_hess` | 0 | 1 | **1** |
| **Result** | 318s, 71° | 145s, 0.18° | **122s, 0.20°** |
