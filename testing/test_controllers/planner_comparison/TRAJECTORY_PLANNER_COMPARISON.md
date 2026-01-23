# Trajectory Planner Comparison Framework

This module provides a comprehensive comparison framework for satellite attitude trajectory planners.

## Overview

The ALTRO (Augmented Lagrangian TRajectory Optimizer) planner is compared against several alternative trajectory planning methods commonly used for spacecraft attitude control:

### Planners Compared

1. **ALTRO** (Baseline) - Augmented Lagrangian iLQR with constraint handling
2. **Eigenaxis + Trapezoidal Velocity** - Industry-standard baseline using eigenaxis rotation with trapezoidal angular velocity profile
3. **Polynomial Trajectory Shaping** - Smooth offline trajectory generation using polynomial interpolation (5th or 7th order)
4. **Direct Collocation (DIRCOL)** - Classical NLP-based trajectory optimization with Hermite-Simpson transcription
5. **Pseudospectral (Gauss-Lobatto)** - Spectral method with exponential convergence for smooth problems
6. **Convex MPC** - Real-time capable convex optimization using linearized dynamics
7. **Sequential Convex Programming (SCP)** - Iteratively solve convex approximations

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Final attitude error (degrees), final angular velocity error (rad/s)
- **Compute Time**: Wall-clock time for trajectory generation
- **Convergence Rate**: Number of iterations / time to converge
- **Constraint Satisfaction**: Maximum constraint violation

### Secondary Metrics
- **Trajectory Smoothness**: Control rate (jerk) magnitude
- **Control Effort**: Total integrated control magnitude
- **Variability**: Standard deviation across multiple runs
- **Scalability**: Performance vs problem size (horizon length, state dimension)

## Usage

```bash
# Run all comparisons
python run_planner_comparison.py

# Run specific test scenarios
python run_planner_comparison.py --scenario small_maneuver

# Quick validation test
python run_planner_comparison.py --quick

# Generate detailed report
python run_planner_comparison.py --report comparison_report.md
```

## Directory Structure

```
planner_comparison/
├── README.md                   # This file
├── __init__.py
├── base_planner.py             # Abstract base class for planners
├── altro_wrapper.py            # Wrapper for existing ALTRO planner
├── eigenaxis_trapezoidal.py    # Eigenaxis rotation + trapezoidal velocity profile
├── polynomial_shaping.py       # Polynomial trajectory shaping (5th/7th order)
├── scp_planner.py              # Sequential Convex Programming (optional)
├── comparison_metrics.py       # Metrics computation and analysis
├── test_scenarios.py           # Standard test scenarios
├── run_planner_comparison.py   # Main comparison runner
└── results/                    # Output directory for results
```

## Planner Details

### 1. ALTRO (Augmented Lagrangian TRajectory Optimizer)
- **Type**: Nonlinear optimal control solver
- **Method**: iLQR with augmented Lagrangian constraint handling
- **Goal Type**: Full 3-DOF quaternion control (via `Fixed_Attitude_Goal`) OR 2-DOF pointing (via `ECI_Goal`)
- **Strengths**: Handles nonlinear dynamics, actuator constraints, optimal control, full physics,
  considers magnetic field variations, reaction wheel momentum management, produces TVLQR feedback gains
- **Weaknesses**: Computationally expensive (~5-60s), requires tuning, may not converge, needs full
  satellite/orbit infrastructure
- **Note**: The C++ planner supports both vector-based (3D) and quaternion-based (4D) goals. When
  passing a 4-element goal vector, it uses `quatcostJacobians` for full attitude control.

### 2. Eigenaxis + Trapezoidal Velocity Profile
- **Type**: Geometric/kinematic trajectory generation
- **Method**: 
  1. Compute eigenaxis (shortest rotation axis) between current and goal attitude
  2. Plan trapezoidal angular velocity profile along eigenaxis
  3. Integrate quaternion kinematics
- **Strengths**: Fast, simple, deterministic, flight-proven
- **Weaknesses**: Doesn't optimize for control effort, ignores dynamics constraints

### 3. Polynomial Trajectory Shaping
- **Type**: Smooth trajectory interpolation
- **Method**:
  1. Define boundary conditions (position, velocity, acceleration at start/end)
  2. Fit polynomial (typically 5th or 7th order) to satisfy boundary conditions
  3. Evaluate polynomial at discrete times
- **Strengths**: Smooth trajectories, no jerk discontinuities, predictable
- **Weaknesses**: May violate constraints, doesn't optimize for dynamics

### 4. Direct Collocation (DIRCOL)
- **Type**: Direct transcription to NLP
- **Method**: Hermite-Simpson collocation with scipy.optimize
- **Strengths**: Robust convergence, handles constraints, widely used in aerospace
- **Weaknesses**: Large NLP problems, slower than specialized solvers

### 5. Pseudospectral (Gauss-Lobatto)
- **Type**: Spectral collocation method
- **Method**: Legendre-Gauss-Lobatto nodes with spectral differentiation
- **Strengths**: Exponential convergence for smooth problems, high accuracy with few nodes
- **Weaknesses**: Struggles with non-smooth solutions (bang-bang), less robust for discontinuities

### 6. Convex MPC
- **Type**: Model Predictive Control with convex optimization
- **Method**: Linearize dynamics, solve QP using OSQP, iterate
- **Strengths**: Fast (convex solvers), real-time capable, handles constraints
- **Weaknesses**: Linearization error, may need multiple iterations, local optimum

### 7. Sequential Convex Programming (SCP)
- **Type**: Convex optimization with successive linearization
- **Method**: Iteratively solve convex subproblems until convergence
- **Strengths**: Handles constraints well, provable convergence properties
- **Weaknesses**: Requires good initial guess, may be slow

## Where ALTRO Shines

ALTRO (Augmented Lagrangian TRajectory Optimizer) has several key advantages:

### 1. **Constraint Handling**
- Uses augmented Lagrangian method for elegant equality/inequality constraint handling
- Better than penalty methods (DIRCOL) which require tuning
- More efficient than interior point methods for trajectory optimization

### 2. **Feedback Gains as Byproduct**
- Naturally produces time-varying LQR (TVLQR) gains during backward pass
- Other methods (DIRCOL, Pseudospectral) require separate gain computation
- Critical for tracking control - no extra computation needed

### 3. **Computational Efficiency**
- O(N) complexity per iteration (exploits Markov structure)
- DIRCOL has O(N³) for dense NLP, O(N) with sparse solvers but more overhead
- Pseudospectral requires dense matrix operations

### 4. **Warm Starting**
- iLQR structure allows efficient warm-starting from previous solutions
- Important for replanning and MPC applications
- DIRCOL/Pseudospectral can warm-start but less naturally

### 5. **Full Nonlinear Dynamics**
- Handles complete nonlinear attitude dynamics
- No linearization error (unlike SCP, Convex MPC)
- Important for large maneuvers and high-fidelity simulation

### When Other Methods May Be Better:

| Scenario | Better Choice | Reason |
|----------|---------------|--------|
| Simple rest-to-rest | Eigenaxis/Polynomial | 1000x faster, no optimization needed |
| Very smooth trajectories | Pseudospectral | Spectral convergence, fewer nodes |
| Real-time replanning | Convex MPC | Guaranteed solve time |
| Poorly initialized problems | DIRCOL | More robust to bad initial guess |
| Embedded systems | TinyMPC/Convex MPC | Lower memory, faster |

## Feature Comparison Matrix

| Feature | Eigenaxis | Polynomial | ConvexMPC | SCP | ALTRO |
|---------|-----------|------------|-----------|-----|-------|
| Full 3-DOF attitude | ✓ | ✓ | ✓ | ✓ | ✓ |
| Underactuated (MTQ-only) | ✗ | ✗ | ~ | ~ | ✓ |
| RW momentum dynamics | ✗ | ✗ | ~ | ✓ | ✓ |
| Desaturation (momentum dump) | ✗ | ✗ | ~ | ~ | ✓ |
| External disturbances | ✗ | ✗ | ✗ | ✗ | ~ |
| Path/pointing constraints | ✗ | ✗ | ✗ | ~ | ✓ |
| TVLQR feedback gains | ✗ | ✗ | ✗ | ✗ | ✓ |
| Solve time (30° slew) | 5ms | 5ms | 250ms | 3s | 4s |

**Legend:** ✓=supported, ✗=not supported, ~=partial/requires setup

### When ALTRO is Essential

1. **Underactuated systems (MTQ-only)**: Only ALTRO properly handles the constraint that
   magnetorquers cannot produce torque along the magnetic field direction.

2. **Momentum management/desaturation**: ALTRO can jointly optimize attitude pointing AND 
   reaction wheel desaturation using the `AM_cost` parameter.

3. **Path constraints**: Keep-out cones (don't point sensor at sun), slew rate limits are
   handled naturally via augmented Lagrangian.

4. **Feedback control**: ALTRO produces TVLQR gains as a byproduct for closed-loop tracking.

## Advanced Test Scenarios

The framework includes scenarios that test ALTRO's unique capabilities:

```python
from test_scenarios import ScenarioLibrary

# Underactuated: MTQ-only (no reaction wheels)
scenario = ScenarioLibrary.create_mtq_only(30.0, horizon=300.0)

# Momentum management: desaturate RWs while slewing
scenario = ScenarioLibrary.create_desaturation(angle_deg=30.0, horizon=180.0)

# Disturbance rejection
scenario = ScenarioLibrary.create_with_disturbance(30.0, horizon=60.0)

# Path constraints: avoid pointing at sun
scenario = ScenarioLibrary.create_pointing_constraint(45.0, keep_out_cone_deg=30.0)

# Reduced attitude: 2-DOF boresight pointing (roll-free)
scenario = ScenarioLibrary.create_reduced_attitude()

# Get all ALTRO showcase scenarios
scenarios = ScenarioLibrary.get_altro_showcase_scenarios()
```

## Benchmark Results (Jan 2026)

| Planner | Solve Time | Error (deg) | Notes |
|---------|-----------|-------------|-------|
| Polynomial-5/7 | 6 ± 5 ms | 0.000 | Fastest, perfect for simple slews |
| Eigenaxis+Trap | 7 ± 4 ms | 0.002 | Industry baseline |
| ConvexMPC | 287 ± 100 ms | 0.000 | Real-time capable |
| SCP | 4.5 ± 2.7 s | 0.000 | Robust but slow |
| ALTRO | 4-14 s | 0.000-0.04 | Best for constrained problems |

**Note**: ALTRO's Cayley cost (mode 2) has a singularity at 180° rotations. Use mode 0 or 4
for large maneuvers, or chain smaller slews.

## References

1. ALTRO: Howell, T.A., et al. "ALTRO: A Fast Solver for Constrained Trajectory Optimization"
2. Eigenaxis: Wie, B. "Space Vehicle Dynamics and Control", Chapter 7
3. Polynomial Shaping: Junkins, J.L., Turner, J.D. "Optimal Spacecraft Rotational Maneuvers"
4. DIRCOL: Hargraves & Paris (1987) "Direct Trajectory Optimization Using Nonlinear Programming"
5. Pseudospectral: Ross & Fahroo (2004) "Pseudospectral Knotting Methods for Solving Optimal Control Problems"
6. Convex MPC: Kalabic et al. (2017) "MPC for Spacecraft Attitude Control"
7. SCP: Mao, Y., et al. "Successive Convexification for Fuel-Optimal Powered Landing"
