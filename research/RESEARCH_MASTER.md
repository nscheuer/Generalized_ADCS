# Generalized ADCS Control Laws - Research Summary

**Last Updated:** January 23, 2026

---

## Research Questions

1. **Torque Allocation:** Best method to recreate commanded torque within actuator bounds
2. **Momentum Desaturation:** How to manage RW momentum without degrading pointing
3. **Goal Conversion:** How to achieve full attitude from reduced (vector alignment) objectives

---

## Executive Summary

### Q1: Torque Allocation

**Answer: LP for direction preservation; QP variants for L2 optimality**

| Method | Direction Error | Best Use Case |
|--------|----------------|---------------|
| **LP** | 0° (exact) | Safety-critical, Lyapunov stability proof |
| **QP** | Variable | Maximum torque magnitude |
| **QP + Sign Constraint** | Per-axis | Best overall closed-loop |

**Key Findings:**
- LP preserves direction via equality constraint `τ = α·τ̂_des`
- QP can have 30-60° direction errors in underactuated systems
- QP with physics constraints (damping preservation) achieves best closed-loop performance
- **Critical:** QP requires scaling fix (SCALE=1e6) due to ill-conditioned A matrix

### Q2: Momentum Desaturation

**Answer: Torque-free desaturation is possible but config-dependent**

| Configuration | Torque-Free? | Notes |
|--------------|--------------|-------|
| 3MTQ + 3RW | ✓ Full | MTQ and RW torques can cancel exactly |
| 3MTQ + 1RW | Partial | Only when B ⊥ RW axis (geometric constraint) |
| 4RW only | ✗ | No external torque mechanism |

**Key Insight:** For 3MTQ+1RW, only desaturate when B-field geometry is favorable.

### Q3: Goal Conversion

**Answer: Multi-vector tracking or alternating**

| Method | Full Attitude? | Notes |
|--------|---------------|-------|
| Multi-vector | ✓ | Track 2 body vectors to 2 inertial targets |
| Alternating | ✓ | Switch between vectors each timestep |
| Single vector | ✗ | 1 DOF unconstrained (axial rotation) |

**Key Insight:** Alternating every timestep ≈ multi-vector (inertia low-pass filters).

---

## Detailed Findings

### LP vs QP Allocation

#### Why LP Preserves Direction

The LP formulation:
```
max  α
s.t. A·u = α·τ̂_des   ← EQUALITY constraint
     lb ≤ u ≤ ub
     α ≥ 0
```

Forces τ to be exactly parallel to τ_des. Direction error = 0° always.

#### Why QP Can Fail

The naive QP:
```
min  ||τ - τ_des||²
s.t. lb ≤ u ≤ ub
```

Minimizes Euclidean distance, not angular error. In underactuated systems, the closest achievable τ may point in a very different direction.

#### QP Scaling Issue

The combined allocation matrix A (RW + MTQ) spans 5 orders of magnitude:
- RW torques: ~1e-3 Nm
- MTQ dipoles: ~0.4 Am²
- B-field effects: ~1e-5 T

**Solution:** Scale the objective by 1e6 to improve conditioning.

### Physics-Based QP Constraints

Tested 10 physics-based constraints for QP allocation:

| Constraint | Principle | Result |
|------------|-----------|--------|
| Power bound | ω'τ ≤ 0 | **Hurts convergence** |
| Global Lyapunov | V̇ ≤ 0 | Gets stuck |
| Sign preservation | sign(τ) = sign(τ_des) | **Works well** |
| Per-axis Lyapunov | V̇_i ≤ 0 | Too restrictive |
| **1a-Power brake only** | If P_des < 0: ω'τ ≤ 0 | **3.11° - Good!** |
| **3b-Sign critical** | Sign on large axes only | **2.35° - Best!** |

**Critical Insight:** Energy constraints hurt during convergence because the controller needs to inject energy to maintain tracking. Only apply power constraints when the controller is actively braking (P_des < 0).

### Why Pure Lyapunov Constraints Fail

During convergence (θ > 0, ω < 0):
- Controller wants τ > 0 to slow the approach rate
- But P_des = ω'τ_des < 0 (negative power)
- Constraint ω'τ ≤ 0 prevents positive torque
- System can't brake properly → overshoots

**Solution:** Only constrain when controller intends to brake, not based solely on energy.

### Recommended Constraints

#### For General Use: "1a-Power Brake Only"
```python
P_des = omega @ tau_des
if P_des < -epsilon:
    constraints.append(omega @ tau <= 0)
```
Simple, effective, 3.11° final error.

#### For Maximum Performance: "3b-Sign Critical"
```python
threshold = 0.1 * norm(tau_des)
for i in range(3):
    if abs(tau_des[i]) < threshold:
        continue
    if omega[i] > 0 and tau_des[i] < 0:
        constraints.append(tau[i] <= 0)
    elif omega[i] < 0 and tau_des[i] > 0:
        constraints.append(tau[i] >= 0)
```
Ties unconstrained QP at 2.35°.

---

## Controller Comparison Results

### Real Orbit Test (1000s with time-varying B-field)

| Metric | LP | QP | Winner |
|--------|----|----|--------|
| Mean Final Error | **2.48°** | 5.52° | LP |
| Max Final Error | **13.01°** | 31.01° | LP |
| Mean Steady-State | 3.35° | 4.65° | LP |
| Convergence | 5/6 | 5/6 | Tie |
| Head-to-head | **2** | 1 | LP |

### Constant B-field Test (500s, 12 configurations)

| Metric | LP | QP | Winner |
|--------|----|----|--------|
| Mean Error | 3.12° | **2.70°** | QP |
| Max Error | **16.84°** | 21.42° | LP |
| Convergence | 9/12 | **11/12** | QP |

### Key Takeaways

1. **LP is more robust** with time-varying B-fields (real orbits)
2. **QP can achieve better results** with constant/favorable geometry
3. **LP has lower worst-case error** (more predictable)
4. **QP converges more often** but can fail catastrophically
5. **Convergence is actuator-limited** - failures are insufficient authority, not allocator choice

---

## Validated Configurations

| Configuration | Result |
|--------------|--------|
| 3MTQ + 1RW | Partially controllable |
| 3MTQ + 3RW | Full 3-axis control |
| 4RW Pyramid | Full control, no desat |
| LEO various inclinations | Tested |
| Full attitude goals | Multi-vector works |
| Reduced attitude (boresight) | Single vector sufficient |

---

## Rejected Approaches

| Approach | Why Rejected |
|----------|--------------|
| Single weighted QP | Direction errors up to 65° |
| Nullspace desaturation (3RW+MTQ) | Wrong mathematical structure |
| Reachability-aware goal selection | Counterproductive |
| Error-budget continuous desaturation | 3-4× worse pointing |
| Pure Lyapunov QP constraints | Gets stuck, can't converge |

---

## For the Paper

### Include
1. LP formulation with direction preservation proof
2. QP scaling requirement for mixed actuators
3. Physics-based constraint analysis (1a, 3b)
4. Torque-free desaturation formulation
5. Multi-vector attitude conversion

### Exclude
1. Failed constraint variants (detailed failure analysis in code)
2. Nullspace claims for underactuated systems
3. Reachability-aware control (doesn't help)

---

## Code Artifacts

| Purpose | Key Files |
|---------|-----------|
| Allocation comparison | `allocation_comparison.py`, `qp_comprehensive_allocator_test.py` |
| Physics constraints | `qp_physics_constraints.py`, `qp_10_constraints_test.py`, `qp_constraints_revised.py` |
| Controller comparison | `qp_controller_comparison.py` |
| Desaturation | `desaturation_analysis.py`, `continuous_desaturation.py` |
| Goal conversion | `attitude_goal_conversion.py`, `robust_multivector_test.py` |
| Mathematical proofs | `LP_QP_MATHEMATICAL_PROOF.md` (archived) |

---

## Future Work

1. **Adaptive constraint selection** based on state
2. **Real orbit validation** with time-varying B-field
3. **Hardware-in-the-loop** testing
4. **Thruster integration** for rapid slews
5. **Optimal gain scheduling** for underactuated configs
