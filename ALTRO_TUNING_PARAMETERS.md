# ALTRO Tuning Parameters Reference

## Overview
This document lists ALL tunable parameters for the ALTRO trajectory planner.
**Don't try to tune all at once** - pick a few key ones based on the problem.

---

## 1. TIMESTEP SETTINGS
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt_tp` | 30 | Trajectory planning timestep (seconds). Coarser = faster but less accurate. Must give N>=4 points per 60s segment, so dt_tp <= 20. |
| `dt_tvlqr` | 1 | TVLQR feedback timestep (seconds). Usually 1s. |
| `tvlqr_len` | 60 | Length of each TVLQR segment (seconds) |
| `tvlqr_overlap` | 15 | Overlap between TVLQR segments (seconds) |

---

## 2. INITIAL TRAJECTORY (`bdot_on`)
| Value | Description |
|-------|-------------|
| 0 | Random initial trajectory (fastest, may not converge) |
| 1 | B-dot detumble initial guess |
| 2 | B-dot with some modifications |
| 3 | B-dot with quaternion-aware initial guess (often best for convergence) |

---

## 3. COST WEIGHTS - `cost_main` (Main trajectory optimization)

### State Costs
| Parameter | Default | Description |
|-----------|---------|-------------|
| `angle` | 1e3 | Running cost on pointing error |
| `angle_N` | 1e6 | Terminal cost on pointing error |
| `ang_vel` | 1e3 | Running cost on angular velocity magnitude |
| `ang_vel_N` | 1e5 | Terminal cost on angular velocity |
| `ang_vel_mag` | 0 | Running cost on ω alignment with B-field |
| `ang_vel_mag_N` | 0 | Terminal cost on ω alignment with B-field |

### Control Costs
| Parameter | Default | Description |
|-----------|---------|-------------|
| `control_mult` | 1.0 | Multiplier on all control costs. Lower = more aggressive control. |

### Cost Function Type
| Parameter | Default | Description |
|-----------|---------|-------------|
| `ang_cost_func_type` | 2 | 0=1-dot, 1=0.5*(1-dot)², 2=acos(dot), 3=0.5*acos²(dot) |
| `use_full_cost_hessian` | True | Use full Newton Hessian (True) or Gauss-Newton approx (False) |
| `use_raw_control_cost` | False | If True: 0.5*u'Wu. If False: 0.5*(u-u_prev)'W(u-u_prev) |

---

## 4. COST WEIGHTS - `cost_tvlqr` (TVLQR tracking gains)
Same parameters as `cost_main`, but used for computing TVLQR feedback gains.
Typically higher `ang_vel` costs for tighter tracking.

---

## 5. COST WEIGHTS - `cost_second` (Second pass refinement)
Same parameters as `cost_main`. Used in pass2 if different costs desired.

---

## 6. ACTUATOR CONTROL WEIGHTS (on PlannerSettings directly)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `mtq_control_weight` | 1e3 | Base weight for MTQ control cost |
| `rw_control_weight` | 1e7 | Base weight for RW control cost. Higher = prefer MTQs over RW. |
| `rw_AM_weight` | 1e4 | Cost on RW angular momentum (prevents saturation) |
| `rw_stic_weight` | 1e0 | Cost on low RW speeds (avoids stiction region) |

**Note:** Actual cost = `base_weight * actuator_weight / (max_torque²)` 
With RW max_torque=0.0023 Nm: effective_cost = 1e7 / 5.29e-6 ≈ 1.89e12

---

## 7. PASS 1 CONVERGENCE SETTINGS (`ps.pass1.convergence`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_outer_iter` | 10 | Max augmented Lagrangian outer iterations |
| `max_inner_iter` | 50 | Max iLQR inner iterations per outer |
| `max_iter` | 500 | Absolute max total iterations |
| `grad_tol` | 1e-4 | Gradient norm tolerance for convergence |
| `ilqr_cost_tol` | 1e-4 | Cost change tolerance within iLQR |
| `cost_tol` | 1e-4 | Cost change tolerance for outer loop |
| `z_count_lim` | 10 | Max consecutive small changes before stopping |
| `cmax` | 1e-3 | Max constraint violation for convergence |
| `max_cost` | 1e10 | Abort if cost exceeds this |
| `xmax` | 1e6 | Abort if state exceeds this |

---

## 8. PASS 1 AUGMENTED LAGRANGIAN (`ps.pass1.aug_lag`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `penalty_init` | 100 | Initial penalty parameter μ |
| `penalty_max` | 1e8 | Maximum penalty |
| `penalty_scale` | 10 | Penalty multiplier each outer iteration |
| `lambda_max` | 1e8 | Max Lagrange multiplier magnitude |

---

## 9. PASS 1 REGULARIZATION (`ps.pass1.regularization`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_dynamics_hess` | 1 | Include dynamics Hessians (0=no, 1=yes) |
| `use_constraint_hess` | 0 | Include constraint Hessians (0=no, 1=yes) |

---

## 10. PASS 2 SETTINGS
Same structure as Pass 1: `ps.pass2.convergence`, `ps.pass2.aug_lag`, `ps.pass2.regularization`
Pass 2 typically has fewer iterations (refinement only).

---

## 11. INITIAL TRAJECTORY CONFIG (`ps.init_traj`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bdot_gain` | 500 | B-dot controller gain for initial trajectory |
| `quat_gain` | varies | Quaternion feedback gain |
| `vel_gain` | varies | Velocity feedback gain |

---

## 12. DISTURBANCE PLANNING
| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_gg` | False | Plan for gravity gradient torque |
| `include_srp` | False | Plan for solar radiation pressure |
| `include_drag` | False | Plan for atmospheric drag |
| `include_resdipole` | False | Plan for residual magnetic dipole |
| `include_prop` | False | Plan for propellant slosh |
| `include_gendist` | False | Plan for general disturbance torque |

---

## Quick Tuning Guide

### Problem: Trajectory doesn't converge to goal
1. Increase `angle_N` (terminal angle cost)
2. Try `bdot_on=3` for better initial guess
3. Increase `max_outer_iter` and `max_inner_iter`
4. Decrease `rw_control_weight` to allow more RW use

### Problem: Trajectory oscillates
1. Increase `ang_vel` and `ang_vel_N` (velocity costs)
2. Increase `control_mult` to penalize control changes
3. Try `use_raw_control_cost=False` for smoother control

### Problem: Planner is too slow
1. Increase `dt_tp` (coarser timestep)
2. Decrease `max_outer_iter` and `max_inner_iter`
3. Use `bdot_on=0` (random init, faster but may not converge)
4. Decrease terminal costs (less work to converge)

### Problem: TVLQR tracking diverges
1. Check `cost_tvlqr` weights - may need higher `ang_vel` 
2. Verify `ang_cost_func_type=2` in cost_tvlqr
3. Ensure planned trajectory is smooth (no oscillations)

### Problem: Planner hangs/very slow
1. Check if `control_mult` or control weights are too low (numerical issues)
2. Check if costs are too unbalanced (e.g., 1e10 vs 1e0)
3. Try coarser `dt_tp`
4. Reduce iteration limits to see if it's just slow or stuck

---

## Current Working Settings (from earlier tests)
```python
ps = PlannerSettings(est_sat=real_sat, bdot_on=0, dt_tp=10, dt_tvlqr=1)
ps.mtq_control_weight = 1e2
ps.rw_control_weight = 1e2
ps.cost_main = CostWeights(
    angle=1e4, angle_N=1e6,
    ang_vel=1e7, ang_vel_N=1e9,  # High velocity costs for smooth trajectories
    control_mult=0.1
)
# This gave smooth planned trajectories but TVLQR tracking diverged
```

---

## bdot_on Comparison Results (60s trajectory, BC2 3+1)

| bdot_on | Plan Time | Trajectory Quality | Final ω |
|---------|-----------|-------------------|---------|
| 0 | 2.8s | Oscillates, finds high-velocity pass-through | 20°/s |
| **1** | 4.5s | **Best monotonic convergence (low iterations)** | **1.4°/s** |
| 2 | 5.3s | Oscillates back up past initial | 20°/s |
| 3 | 3.6s | Goes past 90°, high velocity solution | 18°/s |

**Key Finding**: With low iterations (4 outer, 20 inner), `bdot_on=1` gives the best results:
- Monotonic error reduction: 72° → 51° → 37° → 16°
- Low final angular velocity: 1.4°/s
- Reasonable planning time: 4.5s

**Warning**: More iterations can make things WORSE! The optimizer may find oscillatory local minima that pass through the goal with high velocity instead of converging smoothly.

### Recommended Settings for BC2 3+1
```python
ps = PlannerSettings(est_sat=sat, bdot_on=1, dt_tp=10, dt_tvlqr=1)
ps.pass1.convergence.max_outer_iter = 4  # Low iterations work better!
ps.pass1.convergence.max_inner_iter = 20
ps.pass2.convergence.max_outer_iter = 2
ps.pass2.convergence.max_inner_iter = 10
```
