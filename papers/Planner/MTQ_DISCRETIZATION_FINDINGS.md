# MTQ-Only Discretization & Pass2 Settings: Comprehensive Study

## Problem Statement

For MTQ-only systems, the two-pass optimization (coarse Pass1, fine Pass2) struggles because:
1. MTQ torque = m × B depends on instantaneous B-field direction
2. B-field varies significantly within coarse timesteps
3. Controls interpolated from coarse grid become infeasible on fine grid

## Solution: Interpolate States + Solve-for-Controls

### Algorithm
1. **Pass 1**: Optimize on coarse grid (dt=20s) → Find good trajectory shape
2. **Transition**: 
   - Interpolate states (cubic) to fine grid
   - Solve for MTQ controls in **body frame**: `m = (B × τ) / |B|²`
   - Clamp controls to limits
   - Forward propagate with clamped controls
3. **Pass 2**: Refine on fine grid (dt=2s) with modified cost weights

### Key Insight: Reduce Angle Weight for Pass 2

**0.1x angle weight** gives best results:
- 90% success rate across random seeds
- Mean error: 2.42° (most cases < 1°)
- Allows constraints to drive solution rather than fighting angle cost

## Results Summary

### Direct 20s → 2s (with solve-for-controls transition)

| Config | Success Rate | Mean Error (°) | Mean cmax |
|--------|--------------|----------------|-----------|
| **0.1x angle** | **90%** | **2.42** | 8.59e-02 |
| 10x ctrl + 0.2x angle | 100% | 13.72 | 5.40e-01 |
| 0.2x angle | 60% | 1.45 | 2.37e-01 |
| baseline | 30% | 22.07 | 7.63e-01 |
| 10x ctrl | 10% | 3.08 | 4.19e-01 |

### Gradual Refinement (20s → 10s → 5s → 2s)

| Metric | Value |
|--------|-------|
| Success Rate | **100%** |
| Mean Time | 58.9s |
| Mean Error | 2.70° |
| Mean cmax | 9.71e-01 |

**Gradual refinement** gives 100% robustness but higher cmax values.

## Recommendations

### For MTQ-Only Systems

1. **Use coarse Pass1**: dt=20s works well for finding trajectory shape
2. **Solve-for-controls transition**: Transform to body frame, use `m = (B × τ) / |B|²`
3. **Reduce angle weight 10x for Pass2**: Let constraints drive refinement
4. **If robustness is critical**: Use gradual refinement (20s→10s→5s→2s)

### Implementation

```python
# Pass 2 cost modification for MTQ-only
cost_mods = {
    0: base_cost[0] * 0.1,  # angle_weight
    5: base_cost[5] * 0.1,  # angle_weight_N (terminal)
}
```

### Why This Works

1. **Constraints include goal pointing**: Lower angle cost doesn't abandon the goal
2. **Avoids competing gradients**: High angle cost fights constraint penalties
3. **Better warm start utilization**: solve-for-controls is already close to optimal

## Files

- Experiment scripts: `/tmp/pass2_experiments*.log`
- Results: `/tmp/pass2_experiments*_results.md`
- Plots: `/tmp/pass2_*.png`, `/tmp/gradual_*.png`

## Date

Generated: 2026-02-02
