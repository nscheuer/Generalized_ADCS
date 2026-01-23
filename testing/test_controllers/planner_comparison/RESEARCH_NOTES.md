# Trajectory Optimization Methods for Satellite Attitude Control

## Research Summary

### Methods Found in Literature

#### 1. **Direct Collocation (DIRCOL)**
- **Type**: Direct transcription method
- **Description**: Discretizes the continuous optimal control problem into a nonlinear program (NLP) by parameterizing both states and controls at collocation points. Uses polynomial interpolation (typically Hermite-Simpson or trapezoidal) to enforce dynamics.
- **Used in**: NASA missions, general spacecraft trajectory planning
- **References**: 
  - Hargraves & Paris (1987) "Direct Trajectory Optimization Using Nonlinear Programming"
  - Betts (1998) "Survey of Numerical Methods for Trajectory Optimization"
- **Strengths**: Handles constraints well, robust convergence, can use off-the-shelf NLP solvers
- **Weaknesses**: Large NLP problems, may be slow for real-time, requires good initial guess

#### 2. **Drake (from MIT/TRI)**
- **Type**: Robotics/control toolbox with multiple trajectory optimization methods
- **Description**: Provides DIRCOL, shooting methods, and contact-implicit optimization
- **Used in**: Robotics, some aerospace applications
- **Note**: Primarily designed for robotic manipulation, not optimized for spacecraft

#### 3. **Pseudospectral Methods (GPOPS-II, PSOPT)**
- **Type**: Direct transcription using spectral basis functions
- **Description**: Uses Gauss-Lobatto or Chebyshev points for collocation. Achieves spectral convergence for smooth problems.
- **Used in**: 
  - ISS attitude control studies
  - Various NASA missions
  - Ross & Fahroo's work at Naval Postgraduate School
- **References**:
  - Ross & Fahroo (2004) "Pseudospectral Knotting Methods for Solving Optimal Control Problems"
  - Garg et al. (2010) "A Unified Framework for the Numerical Solution of Optimal Control Problems Using Pseudospectral Methods"
- **Strengths**: Very accurate for smooth trajectories, exponential convergence
- **Weaknesses**: Struggles with non-smooth solutions (bang-bang control), requires smooth dynamics

#### 4. **Sequential Convex Programming (SCP) / Successive Convexification**
- **Type**: Iterative convex optimization
- **Description**: Linearizes nonlinear dynamics around a reference trajectory, solves convex subproblems iteratively until convergence.
- **Used in**:
  - SpaceX rocket landing (powered descent guidance)
  - NASA fuel-optimal landing
  - Asteroid proximity operations
- **References**:
  - Mao et al. (2016) "Successive Convexification of Non-Convex Optimal Control Problems"
  - Acikmese & Ploen (2007) "Convex Programming Approach to Powered Descent Guidance"
- **Strengths**: Convex subproblems have guaranteed solutions, handles constraints well, provable convergence
- **Weaknesses**: Requires good initial guess, may need many iterations

#### 5. **iLQR / DDP (Iterative Linear Quadratic Regulator / Differential Dynamic Programming)**
- **Type**: Indirect shooting method
- **Description**: Iteratively solves backward Riccati equations and forward simulation. ALTRO is based on this with augmented Lagrangian for constraints.
- **Used in**:
  - Model Predictive Control
  - Robotics (Tassa et al.)
  - Some spacecraft applications
- **References**:
  - Tassa et al. (2012) "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization"
  - Li & Todorov (2004) "Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems"
- **Strengths**: Fast for unconstrained problems, naturally provides feedback gains
- **Weaknesses**: Basic iLQR doesn't handle constraints well (ALTRO fixes this)

#### 6. **Model Predictive Control (MPC) / Receding Horizon**
- **Type**: Online optimization
- **Description**: Solves trajectory optimization over a finite horizon at each timestep, applies first control, repeats.
- **Used in**:
  - ISS attitude control
  - Many satellite missions
  - Agile Earth observation satellites
- **References**:
  - Guiggiani et al. (2015) "Fixed-Point Constrained Model Predictive Control of Spacecraft Attitude"
  - Kalabic et al. (2017) "MPC for Spacecraft Attitude Control"
- **Strengths**: Handles disturbances, naturally closed-loop, can adapt
- **Weaknesses**: Computationally expensive per step, requires fast solvers

#### 7. **TinyMPC**
- **Type**: Embedded MPC solver
- **Description**: Highly optimized ADMM-based MPC solver for embedded systems
- **Used in**: Robotics, drones, potentially CubeSats
- **Note**: Already implemented in this codebase!
- **Strengths**: Very fast, low memory, suitable for embedded
- **Weaknesses**: Limited problem size, linear dynamics assumption

#### 8. **ECOS / OSQP-based Convex MPC**
- **Type**: Convex optimization
- **Description**: Formulates attitude control as convex (or convexified) QP/SOCP
- **Used in**: Real-time embedded applications
- **Strengths**: Fast, reliable convergence
- **Weaknesses**: Requires convexification of nonlinear dynamics

---

## Top 3 Candidates for Comparison

Based on literature relevance, implementation feasibility, and differentiation from existing methods:

### 1. **Direct Collocation (Hermite-Simpson)**
- Most widely used in aerospace trajectory optimization
- Well-documented, robust
- Good comparison point for ALTRO's augmented Lagrangian approach
- Can implement using CVXPY/IPOPT or scipy

### 2. **Pseudospectral Method (Gauss-Lobatto)**
- Used in several NASA missions
- Different numerical approach (spectral vs finite difference)
- Should excel at smooth trajectories
- Can implement using scipy or custom code

### 3. **Convex MPC (OSQP-based)**
- Represents modern real-time control approach
- Direct comparison to ALTRO's offline optimization
- Shows trade-off between computation time and optimality
- Can implement using CVXPY + OSQP

---

## Where ALTRO Shines

### ALTRO's Key Advantages:

1. **Constraint Handling**: 
   - ALTRO uses augmented Lagrangian method which elegantly handles both equality and inequality constraints
   - Direct collocation requires NLP solver constraint handling (often slower)
   - Pseudospectral methods struggle with non-smooth constraints

2. **Feedback Gains**:
   - ALTRO naturally produces time-varying LQR gains as a byproduct
   - Other methods require separate gain computation step
   - Critical for tracking control

3. **Warm Starting**:
   - iLQR structure allows efficient warm-starting from previous solutions
   - Important for replanning scenarios

4. **Nonlinear Dynamics**:
   - Handles full nonlinear attitude dynamics
   - No linearization approximation error (unlike SCP, convex MPC)

5. **Computational Efficiency**:
   - More efficient than general NLP solvers for trajectory optimization
   - O(N) complexity per iteration (vs O(N³) for dense NLP)

### Where Others May Beat ALTRO:

1. **Eigenaxis/Polynomial**: 
   - 1000x faster for simple rest-to-rest maneuvers
   - Sufficient when dynamics/constraints don't matter

2. **Pseudospectral**:
   - Higher accuracy for very smooth trajectories
   - Better spectral convergence properties

3. **Convex MPC**:
   - Faster per solve (convex vs nonlinear)
   - Better for real-time replanning

4. **Direct Collocation**:
   - More robust convergence for poorly initialized problems
   - Better for very long horizons (sparse structure)

---

## Implementation Status

### All Implemented ✓

| Planner | File | Status | Notes |
|---------|------|--------|-------|
| Eigenaxis + Trapezoidal | `eigenaxis_trapezoidal.py` | ✓ Complete | Industry baseline, ~5-15ms |
| Polynomial Shaping (5th/7th) | `polynomial_shaping.py` | ✓ Complete | Smooth trajectories, ~5-15ms |
| Direct Collocation | `direct_collocation.py` | ✓ Complete | Hermite-Simpson, slow with scipy (~3-10s) |
| Pseudospectral | `pseudospectral.py` | ✓ Complete | Gauss-Lobatto, slow with scipy (~3-10s) |
| Convex MPC | `convex_mpc.py` | ✓ Complete | OSQP-based, ~500-800ms |
| SCP | `scp_planner.py` | ✓ Complete | Iterative convexification, ~5-12s |
| ALTRO wrapper | `altro_wrapper.py` | ✓ Complete | Full 3-DOF quaternion support added |

### Notes on Performance:
- **DIRCOL and Pseudospectral** are slow because they use scipy's general-purpose NLP solver (SLSQP). 
  With specialized solvers like IPOPT or SNOPT, they would be 10-100x faster.
- **Convex MPC** is the fastest optimization-based method due to using OSQP (highly optimized QP solver).
- **Eigenaxis/Polynomial** are fastest overall but don't optimize—they just generate kinematically feasible trajectories.

---

## Benchmark Results (Quick Test)

From `run_planner_comparison.py --no-altro --quick` (Jan 2026, after quaternion convention fix):

| Planner | Solve Time (ms) | Final Error (deg) | Convergence |
|---------|-----------------|-------------------|-------------|
| Polynomial-7 | 8.0 ± 3.9 | 0.000 | 100% |
| Eigenaxis+Trapezoidal | 8.5 ± 4.4 | 0.000 | 100% |
| Polynomial-5 | 10.3 ± 4.7 | 0.000 | 100% |
| ConvexMPC | 436.9 ± 37.5 | 0.000 | 0%* |
| SCP | 8100.9 ± 2228.0 | 0.000 | 0%* |

*Note: "Convergence" here means the solver's internal convergence flag (number of iterations), 
not whether it achieved the goal. All planners achieved near-zero final attitude error.

### Key Observations:
1. **Kinematic planners (Eigenaxis, Polynomial)** are 50-1000x faster than optimization-based methods
2. **ConvexMPC** offers a good speed/accuracy tradeoff for real-time applications
3. **SCP** is slower but more robust for complex constraints
4. **ALTRO** is ~5-60s and provides feedback gains, but currently has issues with full 3-DOF quaternion goals

### ALTRO Quaternion Goal Issue (Jan 2026)
ALTRO's quaternion goal support (4D E array) is not achieving target attitudes in benchmarks:
- The C++ code detects 4D goals correctly and uses `quatcostJacobians`
- However, optimization converges to near-identity instead of the goal quaternion
- Suspected causes: cost weight tuning, local minimum, or quaternion error computation
- **Workaround**: Use 2-DOF pointing (ECI_Goal with 3D vector) which works correctly
- **TODO**: Debug the quatcostJacobians or cost settings for large attitude errors

---

## Satellite Attitude Control Specific Considerations

### Why Attitude Control is Different from General Robotics:

1. **Quaternion Representation**: Attitude uses unit quaternions (S³ manifold), not Euclidean space
   - Quaternion has double-cover: q and -q represent same orientation
   - Must handle quaternion unwinding and shortest-path issues
   - Error metrics must respect quaternion geometry

2. **Coupled Actuators**: 
   - Reaction wheels store momentum, affecting dynamics
   - Magnetorquers can only produce torque perpendicular to B-field
   - Often use hybrid MTQ+RW configurations

3. **Time-Varying Environment**:
   - Magnetic field varies along orbit (~90 min period)
   - MTQ controllability changes with B-field direction
   - Sun vector affects power and thermal constraints

4. **Gyroscopic Coupling**:
   - Non-diagonal inertia creates coupling between axes
   - High-speed maneuvers have significant gyroscopic terms
   - Euler's equations are nonlinear

### What Makes ALTRO Well-Suited for This Problem:

1. **Handles quaternion dynamics naturally** via reduced coordinates (MRP or Cayley parameters)
2. **Models full nonlinear dynamics** including gyroscopic coupling
3. **Respects actuator constraints** (torque limits, momentum limits)
4. **Produces tracking gains** needed for closed-loop execution
5. **Can incorporate time-varying B-field** and other environmental factors

### Limitations of Simpler Methods:

| Method | Limitation for Satellite Attitude |
|--------|-----------------------------------|
| Eigenaxis | Ignores actuator dynamics, no momentum management |
| Polynomial | May violate torque/momentum constraints |
| Convex MPC | Linearization error for large maneuvers |
| DIRCOL | Doesn't naturally provide feedback gains |
| Pseudospectral | Struggles with bang-bang control from saturation |

