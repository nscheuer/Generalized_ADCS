# Trajectory Planner Profiling Results

## Summary

The trajectory planner performance was analyzed across different configurations to identify bottlenecks and optimization opportunities.

## Key Findings

### 1. Per-Iteration Timing

| Configuration | N (timesteps) | Mean Iter Time | Time/N |
|--------------|---------------|----------------|--------|
| 60s, dt=1s   | 61            | 42.6 ms        | 0.70 ms/timestep |
| 60s, dt=2s   | 31            | 39.3 ms        | 1.27 ms/timestep |
| 120s, dt=2s  | 61            | 38.4 ms        | 0.63 ms/timestep |

**Observation**: Time per iteration scales approximately linearly with N (sub-linear actually).

### 2. Iteration Breakdown (from per-iteration analysis)

Typical iteration times for N=61 (60s horizon, dt=1s):
- **First iteration**: ~250 ms (includes initialization)
- **Subsequent iterations**: 15-50 ms each
- **Stuck iterations** (with regularization): 100-130 ms

The large variation (15-130 ms) suggests:
- Normal iterations: ~20-40 ms (backward + forward pass)
- Line search failures trigger regularization increase → longer forward pass
- Regularization changes require recomputation → spikes in timing

### 3. Optimization Breakdown

For a typical 90° slew with 60s horizon:
- **Total iterations**: ~45 (3 outer × 15 inner)
- **Total time**: ~2s
- **Time per iteration**: ~44 ms average

Within each iteration:
- `backwardPass`: Riccati recursion, computes K and d
- `forwardPass`: Line search with multiple rollouts
- `cost2Func`: Cost evaluation (called 2x per iteration)
- `maxViol`: Constraint check

### 4. Scaling Behavior

| Duration | N   | Time (s) | Notes |
|----------|-----|----------|-------|
| 30s      | 31  | 3.36     | High per-iter cost for small N |
| 60s      | 61  | 2.78     | Sweet spot |
| 90s      | 91  | 2.34     | Converges faster |
| 120s     | 121 | 2.74     | Still efficient |

**Empirical scaling**: time ~ N^(-0.15) (sub-linear!)

This counter-intuitive result is because:
1. Longer horizons give the optimizer more "room" to find good trajectories
2. Better conditioning with more timesteps
3. Fixed iteration count in tests

## Bottleneck Analysis

### Primary Bottlenecks (in order of impact):

1. **Number of iterations** (most significant)
   - Currently using fixed max_outer=3, max_inner=15
   - Early termination when converged would help
   - Warm-starting from previous trajectory could reduce iterations

2. **Forward pass with line search**
   - Multiple rollouts per iteration when line search doesn't converge
   - Each rollout is O(N) in dynamics evaluations
   - Line search can try up to 20 step sizes

3. **Backward pass (Riccati)**
   - O(N) matrix operations
   - Dense matrices for small state dimension (8)
   - Already well-optimized in C++

4. **Regularization handling**
   - When Hessian isn't positive definite, regularization increases
   - Requires re-running backward pass
   - Can add 50-100ms to an iteration

### Minor Overheads:

- Orbit propagation: ~50-100 ms (one-time per trajectory)
- C++ planner construction: <10 ms
- Python-C++ data transfer: <5 ms per call

## Optimization Recommendations

### High Impact:

1. **Warm-starting** (estimated 2-3x speedup)
   - Use previous trajectory as initial guess
   - Only replan from current time, not full horizon
   - Reduces iterations from ~45 to ~15

2. **Adaptive termination** (estimated 1.5x speedup)
   - Stop inner loop when gradient < tolerance
   - Stop outer loop when constraints satisfied
   - Current code may over-iterate

3. **Reduced line search** (estimated 1.2x speedup)
   - Trust region methods instead of line search
   - Fewer rollouts per iteration

### Medium Impact:

4. **Horizon management**
   - Use coarser dt for exploration (pass 1)
   - Use finer dt for refinement (pass 2)
   - Interpolate between resolutions

5. **Parallel rollouts** (in forward pass)
   - Multiple line search candidates evaluated in parallel
   - Would require C++ changes

### Low Impact (already optimized):

6. **Matrix operations** - Already using Armadillo
7. **Memory allocation** - Pre-allocated in C++
8. **Dynamics evaluation** - Efficient quaternion kinematics

## Comparison: Legacy vs Normalized Settings

| Metric | Legacy | Well-Conditioned |
|--------|--------|------------------|
| Time | 15.80s | 5.25s |
| Error | 0.58° | 0.51° |
| Condition | 100,000 | 37,807 |

**3x speedup** with better conditioning due to:
- Faster convergence (fewer iterations to reach tolerance)
- More stable line search (better Hessian conditioning)
- Reduced regularization needs

## Test Configuration

- Satellite: BeaverCube2 (3 MTQ + 1 RW)
- Goal: 90° quaternion slew
- Horizon: 60-120s
- dt: 1-2s
- Python ALILQR wrapper for timing visibility
