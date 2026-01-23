# Trajectory Planner Comparison Results

Generated: 2026-01-23 14:40:05

## Summary Statistics

| Planner | Solve Time (ms) | Final Error (deg) | Control Effort | Convergence |
|---------|----------------|-------------------|----------------|-------------|
| Eigenaxis+Trapezoidal | 6.9 ± 4.0 | 0.00 ± 0.01 | 0.00 ± 0.00 | 100% |
| Polynomial-5 | 5.6 ± 3.3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 100% |
| Polynomial-7 | 5.5 ± 3.1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 100% |
| SCP | 4994.4 ± 2908.3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0% |

## Detailed Results by Scenario

### ConstrainedActuator_45deg_50pct

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 8.0 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 7.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 7.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 6.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 6.6 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 7.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 7.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 6.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 8.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| SCP | 6664.4 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 6764.3 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 6485.0 | 0.000 | 0.0000 | 0.00 | ✗ |

### LargeAngle_180deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 11.9 | 0.000 | 0.0000 | 0.01 | ✓ |
| Eigenaxis+Trapezoidal | 11.5 | 0.000 | 0.0000 | 0.01 | ✓ |
| Eigenaxis+Trapezoidal | 11.4 | 0.000 | 0.0000 | 0.01 | ✓ |
| Polynomial-5 | 9.4 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 10.0 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 10.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 9.6 | 0.000 | 0.0000 | 0.01 | ✓ |
| Polynomial-7 | 9.6 | 0.000 | 0.0000 | 0.01 | ✓ |
| Polynomial-7 | 9.5 | 0.000 | 0.0000 | 0.01 | ✓ |
| SCP | 9940.8 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 9845.3 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 12593.6 | 0.000 | 0.0000 | 0.00 | ✗ |

### RestToRest_10deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 2.8 | 0.018 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 2.6 | 0.018 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 3.7 | 0.018 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 2.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 2.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 2.0 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 2.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 2.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 2.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| SCP | 2096.7 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 1856.5 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 1873.2 | 0.000 | 0.0000 | 0.00 | ✗ |

### RestToRest_30deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 4.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 4.6 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 3.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 3.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 3.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 3.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 3.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 3.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 3.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| SCP | 3095.6 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 2718.8 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 2962.6 | 0.000 | 0.0000 | 0.00 | ✗ |

### RestToRest_45deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 5.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 5.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 6.6 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 4.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 4.4 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 4.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 4.4 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 4.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 4.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| SCP | 4120.5 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 4242.5 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 6614.6 | 0.000 | 0.0000 | 0.00 | ✗ |

### RestToRest_90deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 15.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 15.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 12.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 11.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 11.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 10.6 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 10.4 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 9.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 10.9 | 0.000 | 0.0000 | 0.00 | ✓ |
| SCP | 7997.8 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 5681.5 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 5663.7 | 0.000 | 0.0000 | 0.00 | ✗ |

### SmallAngle_5deg

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 2.4 | 0.001 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 1.9 | 0.001 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 1.8 | 0.001 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 1.8 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 1.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 1.6 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 1.6 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 1.6 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 1.7 | 0.000 | 0.0000 | 0.00 | ✓ |
| SCP | 1565.3 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 1742.4 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 1618.6 | 0.000 | 0.0000 | 0.00 | ✗ |

### WithRate_45deg_w0.020

| Planner | Time (ms) | Angle Err (deg) | ω Err (rad/s) | Effort | Converged |
|---------|-----------|-----------------|---------------|--------|-----------|
| Eigenaxis+Trapezoidal | 6.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 6.6 | 0.000 | 0.0000 | 0.00 | ✓ |
| Eigenaxis+Trapezoidal | 5.8 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 5.5 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 6.3 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-5 | 6.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 5.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 5.2 | 0.000 | 0.0000 | 0.00 | ✓ |
| Polynomial-7 | 5.1 | 0.000 | 0.0000 | 0.00 | ✓ |
| SCP | 4730.3 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 4584.4 | 0.000 | 0.0000 | 0.00 | ✗ |
| SCP | 4406.4 | 0.000 | 0.0000 | 0.00 | ✗ |
