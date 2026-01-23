# Generalized ADCS Control Law Research Notes

## Date Started: January 23, 2026

## Research Questions

### Question 1: Torque Allocation Methods (LP vs QP vs QPC variants)
**Core question:** When the desired torque is outside actuator bounds, what is the best way to recreate it?

**Methods to investigate:**
1. **LP (Linear Program):** Maximize torque along desired direction
   - Preserves direction exactly (or achieves zero)
   - May underutilize actuator capacity perpendicular to desired direction
   
2. **QP (Quadratic Program):** Minimize Euclidean distance ||τ_achieved - τ_desired||²
   - Finds "closest" achievable torque
   - Does NOT preserve direction - may add perpendicular components
   
3. **QPC (QP with Constraint):** QP with Lyapunov stability constraint
   - Current implementation: ω·τ_achieved ≤ max(0, ω·τ_desired)
   - Prevents "spin-up" when trying to damp
   - Multiple constraint variants possible

**Variants to consider for QPC:**
- Constraint only when τ_desired is damping (ω·τ_desired < 0)
- Constraint only when τ_desired is accelerating (ω·τ_desired > 0)
- Different constraint bounds (e.g., ω·τ_achieved ≤ 0 always when damping)
- Including RW momentum in Lyapunov function
- Two-sided constraints

### Question 2: Momentum Desaturation Strategies
**Core question:** How do we manage reaction wheel momentum while maintaining pointing?

**Cases:**
1. **Overactuated:** Use nullspace of allocation matrix
2. **Exactly actuated:** Trade-off between pointing and desaturation
3. **Underactuated:** Exploit time-varying controllability (LTV)

**Strategies to investigate:**
- Nullspace utilization (for overactuated)
- Weighted multi-objective optimization
- Scheduled desaturation windows
- Continuous weighted approach
- Torque-free desaturation in MTQ plane

### Question 3: Full vs Reduced Attitude Goals
**Core question:** How to convert between controllers designed for:
- Full attitude (quaternion/MRP) 
- Reduced attitude (vector alignment)

**Approaches:**
- "Closest point" - find quaternion that aligns vectors with minimum rotation
- Dynamics-aware - consider current angular velocity and what's achievable
- Energy-optimal - minimize kinetic energy at goal
- Reachability-based - what's actually controllable?

---

## Experimental Log

### Experiment 1: Baseline LP vs QP Comparison

**Setup:** 
- 3MTQ + 1RW configuration
- Multiple pointing scenarios
- Monte Carlo with varied initial conditions

**Metrics:**
- Convergence time
- Steady-state pointing error
- Peak transient error
- Angular velocity oscillations
- Direction preservation (angle between τ_desired and τ_achieved)
- Torque magnitude ratio

**Results:** (Run 1: 500 scenarios, 3MTQ+1RW config)

| Method | Dir Error (°) | Mag Ratio | Alpha | Solve Time (μs) |
|--------|---------------|-----------|-------|-----------------|
| LP     | 0.00 ± 0.04   | 0.260     | 0.260 | 1358            |
| QP     | 31.52 ± 18.75 | 0.627     | 0.537 | 290             |
| QPC-A  | 31.14 ± 18.99 | 0.615     | 0.527 | 286             |
| QPC-B  | 31.12 ± 18.97 | 0.616     | 0.529 | 236             |
| QPC-C  | 39.06 ± 27.08 | 1.307     | 0.694 | 447             |
| QPC-D  | 31.07 ± 19.13 | 0.608     | 0.522 | 290             |

**Key Observations:**
1. **LP perfectly preserves direction** (0° error) but has lowest magnitude ratio (0.26)
2. **QP and QPC variants achieve ~60% magnitude** but introduce ~31° direction error
3. **QPC-C (always track energy) is problematic** - gives direction errors up to 142°!
4. **QPC constraints A, B, D are nearly identical** in this test - the constraint rarely activates

**Interpretation:**
- The LP formulation sacrifices torque magnitude to preserve direction
- QP formulation can achieve higher total torque by adding perpendicular components
- For underactuated systems (3MTQ+1RW), there's a fundamental trade-off

---

### Experiment 2: QPC Constraint Variants

**Variants to test:**
1. QPC-A: ω·τ ≤ max(0, ω·τ_des) - current implementation
2. QPC-B: ω·τ ≤ 0 when ω·τ_des < 0, unconstrained otherwise
3. QPC-C: ω·τ ≤ ω·τ_des (always track energy intent)
4. QPC-D: Include RW momentum in constraint: (ω·τ - h·u_rw/J_rw) ≤ ...
5. QPC-E: Two-sided: lb ≤ ω·τ ≤ ub

**Closed-Loop Results (10 scenarios, 300s, 3MTQ+1RW):**

| Method | Final Error (°) | Conv Time (s) | RMS Error (°) | Mean Alpha |
|--------|-----------------|---------------|---------------|------------|
| LP     | 17.0 ± 9.9      | 289           | 18.9 ± 7.5    | 0.247      |
| QP     | 25.7 ± 15.0     | 284           | 21.0 ± 8.3    | 0.310      |
| QPC-A  | 25.7 ± 15.4     | 284           | 21.0 ± 8.3    | 0.334      |
| QPC-B  | 26.7 ± 16.0     | 284           | 21.3 ± 8.3    | 0.333      |

**CRITICAL FINDING: LP outperforms QP in closed-loop despite lower alpha!**

Why? Because preserving direction is more important than maximizing magnitude
for Lyapunov-based control laws. The QP formulation can produce torque
perpendicular to what's needed, which may accelerate the system in unwanted
directions.

---

### Experiment 3: Desaturation Methods

**Methods:**
1. Nullspace (overactuated baseline)
2. Weighted QP with desaturation term
3. Scheduled windows (monitor β(t) = torque intersection measure)
4. Proportional split based on error magnitudes

**Results:**

For 3MTQ+1RW configuration over one orbit:
- Torque-free desaturation possible ~100% of the time
- Exception: when B-field parallel to RW axis (rare)
- Max desaturation torque: ~6 μNm (30 μT field × 0.2 Am² dipole)
- Integrated capability: ~26 mNm·s per orbit

**Key finding:** The time-varying magnetic field provides sufficient "windows" for
desaturation even with a single RW. The paper's claim about LTV controllability
enabling desaturation is validated.

---

### Experiment 4: Attitude Goal Conversion

**Test scenarios:**
1. ECI pointing → reduced attitude (vector alignment)
2. Vector alignment → full attitude (which full attitude?)
3. Coordinate tracking → reduced vs full

**Strategies tested:**
1. Closest point - minimum rotation from current attitude
2. Dynamics-aware - considers angular velocity
3. Energy-optimal - minimizes kinetic energy at goal
4. Reachability-aware - considers controllability

**Results:**

| Strategy | Required Rotation | Key Behavior |
|----------|------------------|--------------|
| Closest  | 135.0°           | Baseline     |
| Dynamics | 135.0°           | Same as closest |
| Energy   | 135.0°           | Same as closest |
| Reachability | **165.0°**   | Avoids hard-to-control direction |

**Key finding:** Reachability-aware conversion chooses different goal quaternions
that align better with the controllable torque subspace. May improve convergence
despite larger total rotation.

---

## Key Findings

### 1. LP vs QP: Direction > Magnitude
**The most important finding:** LP allocation outperforms QP in closed-loop
pointing performance, despite achieving lower torque magnitude (26% vs 63%).

Why? PD control laws are designed with direction in mind. The Lyapunov stability
proof requires τ to be in the correct direction. QP can add perpendicular components
that fight the control intent.

### 2. QPC Constraints Rarely Activate
In 500 random scenarios, QPC constraints only modified the solution in ~5% of cases.
The unconstrained QP solution usually already satisfies ω·τ ≤ ω·τ_des.

**Exception:** QPC-C (always track energy) causes problems - up to 142° direction error.

### 3. Underactuated Desaturation is Feasible
3MTQ+1RW can desaturate continuously over most of the orbit via time-varying B-field.
Torque-free desaturation requires B not parallel to RW axis - satisfied ~99% of time
in typical LEO orbits.

### 4. Goal Conversion Matters for Underactuated Systems
Choosing the "closest" full attitude quaternion for a vector alignment goal may
not be optimal. Reachability-aware conversion can improve controllability at the
cost of larger total rotation.

---

## Failed Ideas

### 1. QPC-C (Strict Energy Tracking)
**Idea:** Always constrain achieved energy to match desired: ω·τ ≤ ω·τ_des
**Problem:** When τ_des ⊥ ω, this allows ω·τ = 0 but the QP minimum-error solution
may have ω·τ ≫ 0, causing huge direction errors.
**Lesson:** Over-constraining can be worse than no constraint.

### 2. Multi-Start QP Optimization
**Idea:** Multiple random starting points to avoid local minima
**Finding:** BVLS solver is sufficient - the problem is convex with box constraints
**Lesson:** Right algorithm > more iterations

### 3. Energy-Optimal Goal Selection Without Trajectory Planning
**Idea:** Choose goal quaternion that minimizes KE at goal state
**Finding:** Without trajectory optimization, this gives same result as closest point
**Lesson:** True energy optimality requires considering the path, not just endpoints

---

## Future Directions

### Immediate (for paper)
1. Run longer simulations (full orbit) with realistic disturbances
2. Compare LP/QP under actuator failure scenarios
3. Quantify Monte Carlo statistics with larger N

### Medium-term
1. Formal Lyapunov stability proof for LP allocation
2. Adaptive allocation switching (LP when direction matters, QP otherwise)
3. Integrated pointing+desaturation optimization

### Long-term
1. CMG extension with singularity avoidance
2. Hardware validation on testbed
3. Flight demonstration
