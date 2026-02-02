# Mini Monte Carlo Results Summary

## Test Configuration
- **Satellite**: BeaverCube2 (3 MTQ + 1 RW)
- **Slew**: 90° about random axis
- **Horizon**: 120 seconds
- **Cases**: 10 random initial conditions (seeds 1000-1009)

## Results

### Per-Case Results

| Case | Seed | Legacy Time | Legacy Error | Norm Time | Norm Error |
|------|------|-------------|--------------|-----------|------------|
| 1 | 1000 | 11.72s | 0.80° ✓ | 12.93s | 0.53° ✓ |
| 2 | 1001 | 16.87s | 0.58° ✓ | 7.16s | 0.83° ✓ |
| 3 | 1002 | 10.71s | 0.37° ✓ | 28.09s | 0.34° ✓ |
| 4 | 1003 | 11.81s | 1.62° ✓ | 6.29s | 0.12° ✓ |
| 5 | 1004 | 12.76s | 0.75° ✓ | 18.51s | 0.83° ✓ |
| 6 | 1005 | 17.60s | 1.26° ✓ | 12.33s | 0.02° ✓ |
| 7 | 1006 | 14.09s | 6.78° ✗ | 17.89s | 5.64° ✗ |
| 8 | 1007 | 16.39s | 0.31° ✓ | 5.62s | 0.27° ✓ |
| 9 | 1008 | 9.95s | 0.41° ✓ | 10.50s | 0.18° ✓ |
| 10 | 1009 | 7.08s | 0.08° ✓ | 10.06s | 0.17° ✓ |

### Summary Statistics

| Metric | Legacy | Normalized |
|--------|--------|------------|
| Mean Time | 12.90 ± 3.18 s | 12.94 ± 6.57 s |
| Mean Error | 1.30 ± 1.88° | 0.89 ± 1.61° |
| Convergence | 9/10 (90%) | 9/10 (90%) |
| Min Error | 0.08° | 0.02° |
| Max Error | 6.78° | 5.64° |

## Key Observations

1. **Similar convergence rates**: Both achieve 90% success (error < 5°)

2. **Normalized has better accuracy on average**: 0.89° vs 1.30°

3. **High timing variance**: Both show significant case-to-case variation
   - Legacy: 7-18s range
   - Normalized: 5-28s range

4. **Case 7 (seed 1006) fails for both**: This case has a challenging slew axis

5. **No consistent speedup**: Unlike single-case tests, MC shows ~1x speedup
   - Some cases are faster with normalized (cases 2, 4, 8)
   - Some are slower (cases 1, 3, 5)
   - The average is nearly equal

## Why No Consistent MC Speedup?

The 3x speedup seen in single-case tests doesn't appear in MC because:

1. **Problem difficulty varies**: Easy cases converge fast with any settings
2. **Regularization triggers vary**: Hard cases may need more regularization regardless
3. **Initial guess quality**: Random starts have different distances to solution
4. **Slew axis geometry**: Some slew axes align poorly with actuator geometry

## Recommendations

1. **Use normalized settings for new code**: Better accuracy, similar speed
2. **Consider warm-starting**: Would reduce variance significantly
3. **Investigate Case 7 failure**: May indicate actuator geometry issues
4. **For papers**: Report both mean AND variance in timing

## Files Created
- `mini_mc_comparison.py` - Full MC test with 3 configurations
- `mini_mc_nadir.py` - Nadir goal MC test (incomplete due to timeout)
- `profile_planner_bottlenecks.py` - Detailed timing analysis
- `PROFILING_RESULTS.md` - Profiling documentation
