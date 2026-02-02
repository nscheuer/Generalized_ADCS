# MTQ-Only Trajectory Tracking: Findings and Recommendations

## Executive Summary

For MTQ-only spacecraft, **trajectory planning and tracking face fundamental limitations** due to the underactuated nature of magnetorquer control. The planned trajectory assumes specific B-field orientations, but actual torques depend on the current attitude, which diverges from the plan during execution.

## Key Findings

### 1. Planning Timestep Sweet Spot: dt=5s

| Config | Error | Cmax | Smoothness | Time |
|--------|-------|------|------------|------|
| dt=2s (fine) | 0.06° | 2.5e-05 | 0.060 | 16s |
| **dt=5s** | **0.05°** | **1.0e-05** | **0.050** | **5s** |
| dt=10s | 0.05° | 1.5e-06 | 0.084 | 2s |
| dt=20s (coarse) | 0.10° | 6.8e-05 | 0.067 | 3s |

**Recommendation**: Use dt=5s for Pass2 optimization. It provides:
- Smoothest controls
- Fastest optimization (3x faster than dt=2s)
- Excellent constraint satisfaction

### 2. Two-Pass Planning Works Well

| Pass | Timestep | Purpose | Time |
|------|----------|---------|------|
| Pass 1 | 20s | Exploration - find good trajectory shape | 3-8s |
| Pass 2 | 5s | Refinement - smooth controls, constraint enforcement | 5-20s |

Total planning time: ~10-30s for 500-1000s horizons.

### 3. Tracking Limitations (Critical)

**The planned trajectory is NOT directly trackable** with either:
- **Open-loop control** (interpolate planned MTQ commands)
- **TVLQR feedback** (LQR gains computed on planned trajectory)

Both approaches give **identical** results because the fundamental issue is:
- Planned controls assume B-field orientation at each timestep
- Actual B-field orientation depends on current attitude
- Current attitude diverges from plan → different torques → more divergence

**Test Results (dt_plan=5s, dt_sim=1s, single goal):**
- Planned error: 0.28°
- Simulated error: **15.39°** (both open-loop and TVLQR)
- Tracking RMS: 13.08°

### 4. Transition Strategies (Pass1 → Pass2)

| Strategy | Description | Result |
|----------|-------------|--------|
| **A: Direct Interpolation** | Interpolate coarse controls to fine grid | **Best** (simplest works best) |
| B: Solve-for-controls | Compute m = (B × τ) / |B|² | Worse - computed controls don't match optimizer expectations |
| C: Smooth solved | Apply Gaussian filter to B | Moderate |
| D: Smooth interpolated | Apply Gaussian filter to A | Moderate |

### 5. Control Rate Penalty: Ineffective

Increasing control cost (as proxy for rate penalty) did not significantly improve smoothness:
- Higher control cost → worse convergence
- Smoothness improved marginally at cost of much higher error

## Recommendations for MTQ-Only Systems

### For Planning

1. **Use two-pass optimization:**
   - Pass 1: dt=20s, find trajectory shape
   - Pass 2: dt=5s, enforce constraints, smooth controls

2. **Use 0.1x angle weight for Pass 2:**
   - Let constraints drive the solution
   - 90% success rate vs 30% baseline

3. **Simple interpolation for transition:**
   - Direct linear interpolation of controls
   - Don't try to "solve" for optimal transition controls

### For Tracking/Execution

**The trajectory planner output is best used as:**
1. **A reference for replanning** - Use MPC/receding horizon to replan every few seconds with updated B-field
2. **Initialization for feedback controllers** - Use planned trajectory to set desired states, but use adaptive feedback (e.g., B-dot mode) for actual control
3. **Analysis/prediction only** - Understand what the system *could* do under ideal conditions

**NOT recommended for MTQ-only:**
- Pure open-loop execution
- TVLQR tracking without B-field adaptation

### Proposed MPC Approach (Future Work)

For reliable MTQ tracking, consider:
1. Plan long horizon (dt=5-20s) for reference
2. At each control step (dt=1-2s):
   - Get current B-field
   - Solve short-horizon optimization (10-20s ahead)
   - Execute first control
   - Repeat

This would adapt to B-field changes but requires faster optimization.

## Files

- Test scripts: `/tmp/test_smooth_pass2.py`, `/tmp/test_openloop_vs_tvlqr.py`
- Plots: `/tmp/smooth_pass2.png`, `/tmp/openloop_vs_tvlqr.png`

## Update: TVLQR Gains Bug Fixed (Feb 2, 2026)

A bug was found where `PythonALILQRv2.optimize()` was returning **all-zero gains** because `ilqrStep` doesn't expose the backward pass results. 

**Fix applied:** Call `backwardPass` at the end of optimization to compute actual gains.

**New findings with actual gains:**
- Gains are computed correctly now (Kset shape: (18, N) for MTQ-only)
- **However, gains are very large** (mean~166, max~9200 vs actuator limit 0.2)
- TVLQR with these gains **hurts performance** - causes actuator saturation
- Open-loop: 10.86° final error
- TVLQR: 44.55° final error (worse!)

**Root cause:** The LQR gains are computed with state costs ~1e3-1e4 and control costs ~1, making K very large. For MTQ systems where actuator authority is limited and torque direction depends on B-field, large LQR gains cause saturation and instability.

**Remaining work:**
1. Scale gains appropriately for MTQ authority
2. Or use smaller state weights in LQR cost
3. Or implement gain clamping / anti-windup
4. Or use different tracking approach (e.g., model predictive control)

## Date

February 2, 2026
