# Trajectory Planner Comparison Results

Generated: 2026-01-23 14:34:28

## Summary Statistics

| Planner | Solve Time (ms) | Final Error (deg) | Control Effort | Convergence |
|---------|----------------|-------------------|----------------|-------------|
| Eigenaxis+Trapezoidal | 9.8 ± 5.3 | 0.00 ± 0.01 | 0.00 ± 0.00 | 100% |
| Polynomial-5 | 7.5 ± 3.9 | 0.00 ± 0.00 | 0.00 ± 0.00 | 100% |
| Polynomial-7 | 8.5 ± 4.4 | 0.00 ± 0.00 | 0.00 ± 0.00 | 100% |

## Detailed Results by Scenario

### ConstrainedActuator_45deg_50pct

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 15.4 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 10.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 10.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 10.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 10.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 9.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 11.0 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 12.0 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 10.6 | 0.000 | 0.0000 | 0.00 | ✓ |

### LargeAngle_180deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 18.0 | 0.000 | 0.0000 | 0.01 | ✓ |
| Eigenaxis+Trapezoidal | 20.5 | 0.000 | 0.0000 | 0.01 | ✓ |
| Eigenaxis+Trapezoidal | 19.7 | 0.000 | 0.0000 | 0.01 | ✓ |
| Polynomial-5 | 13.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 13.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 14.6 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 17.4 | 0.000 | 0.0000 | 0.01 | ✓ |
| Polynomial-7 | 16.3 | 0.000 | 0.0000 | 0.01 | ✓ |
| Polynomial-7 | 16.5 | 0.000 | 0.0000 | 0.01 | ✓ |

### RestToRest_10deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 4.0 | 0.018 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 6.1 | 0.018 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 6.1 | 0.018 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 3.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 3.6 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 3.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 4.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 6.8 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 2.9 | 0.000 | 0.0000 | 0.00 | ✓ |

### RestToRest_30deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 5.0 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 5.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 7.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 4.4 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 4.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 4.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 4.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 14.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 10.4 | 0.000 | 0.0000 | 0.00 | ✓ |

### RestToRest_45deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 19.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 7.0 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 9.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 7.4 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 9.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 9.8 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 6.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 6.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 6.1 | 0.000 | 0.0000 | 0.00 | ✓ |

### RestToRest_90deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 11.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 11.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 10.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 13.8 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 9.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 8.6 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 9.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 9.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 8.9 | 0.000 | 0.0000 | 0.00 | ✓ |

### SmallAngle_5deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 4.7 | 0.001 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 3.0 | 0.001 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 2.4 | 0.001 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 2.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 2.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 1.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 3.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 2.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 2.0 | 0.000 | 0.0000 | 0.00 | ✓ |

### WithRate_45deg_w0.020

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 9.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 6.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 9.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 6.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 5.4 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 5.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 8.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 5.8 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 8.2 | 0.000 | 0.0000 | 0.00 | ✓ |
