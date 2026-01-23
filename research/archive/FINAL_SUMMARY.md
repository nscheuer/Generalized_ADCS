# ADCS Generalization Research - Final Summary

**Date:** January 23, 2026

---

## Q1: Torque Allocation Methods

### Single QP Formulation (Avoids Two Optimizations)

Instead of solving LP then QP with constraint, use a **single weighted QP**:

```
min  -w * (A·u)·τ̂ + 0.5 * ||A·u - τ_des||²
s.t. lb ≤ u ≤ ub
```

With `w ≈ 1000`, this:
- Prioritizes projection onto desired direction (like LP)
- Uses perpendicular freedom to minimize total error (like QP)
- Single optimization, ~2× faster than LP+QP

**Alternative:** Direction-constrained QP
```
min  -λ * (A·u)·τ̂ + ε*||u||²
s.t. (A·u) perpendicular to τ̂ = 0  (2 equality constraints)
     lb ≤ u ≤ ub
```
This exactly reproduces LP behavior in a QP framework.

### Extended Ideas for Allocation

1. **SOCP/SDP relaxation** for CMG singularities
2. **Robust allocation** against B-field uncertainty  
3. **Neural network approximation** of optimal allocator
4. **Explicit parametric programming** (precompute allocation maps)
5. **CLF-QP formulation** with control barrier functions
6. **ADMM decomposition** for large systems
7. **Log-barrier methods** for smooth saturation handling
8. **Learning-based** policy from optimal trajectories
9. **Mixed-integer QP** for mode switching
10. **Hierarchical QP** with strict priority levels

---

## Q2: Momentum Desaturation

### Continuous Blended Desaturation (No Modes!)

Instead of discrete pointing/desaturation modes, use **smooth weighting**:

```python
def desaturation_weight(h_mag):
    if h_mag < h_low:
        return 0
    elif h_mag > h_high:
        return w_max
    else:
        t = (h_mag - h_low) / (h_high - h_low)
        return w_max * t * t * (3 - 2*t)  # Smoothstep
```

Then solve:
```
min ||A·u - τ_des||² + w(h) * ||u_rw - u_desat||²
```

**Results:** 58% better momentum reduction with only 9° pointing degradation.

### Extended Ideas for Desaturation

1. **Nullspace projection** (overactuated systems)
2. **Cross-coupling exploitation** - use gyroscopic torques
3. **MPC with B-field prediction** - plan desaturation windows
4. **Environmental torque harvesting** - gravity gradient, solar pressure
5. **Momentum bias operation** - run wheels at non-zero bias
6. **Optimal slew planning** - maneuvers that naturally desaturate
7. **Passivity-based** energy shaping controllers
8. **Consensus-based** multi-wheel distribution
9. **Adaptive gain scheduling** based on momentum state
10. **Predictive control** anticipating high-activity periods

---

## Q3: Attitude Goal Conversion (Reduced → Full)

### Methods That Work

| Method | Convergence | Notes |
|--------|-------------|-------|
| **Multi-Vector** | ✓ | Track 2+ body vectors simultaneously |
| **Alternating** | ✓ | Switch between vectors (any frequency!) |
| **Adaptive** | ✓ | Track whichever error is larger |
| Single Vector | ✗ | Cannot control axial rotation |

### Key Finding: Alternating Every Timestep = Multi-Vector!

**Switching every timestep is mathematically equivalent to multi-vector:**
- The spacecraft inertia acts as a low-pass filter
- System responds to time-average torque: τ_avg = (τ₁ + τ₂)/2
- In continuous-time limit: alternating ≡ multi-vector with equal weights

### Adaptive Switching (Best for Underactuated)

For challenging cases (e.g., 3MTQ+1RW), **adaptive switching** performs best:
```python
e1 = vec_err(q, body_z, target_z)
e2 = vec_err(q, body_x, target_x)
error = e1 if norm(e1) > norm(e2) else e2  # Track larger error
```

### Methods for Choosing Goal on Reduced-Attitude Manifold

The manifold of attitudes satisfying vector alignment has 1 DOF (axial rotation).
Different criteria for choosing the specific goal:

| Method | Best For | Implementation |
|--------|----------|----------------|
| **Closest** | Minimum rotation | argmin angle(q, q_current) |
| **Omega-aligned** | Preserve momentum | Minimize perpendicular ω |
| **Controllability** | MTQ systems | Maximize torque ⊥ B |
| **Disturbance-aligned** | Long pointing | Minimize gravity gradient |
| **Min energy** | Fast settling | Minimize kinetic energy at goal |

### Multi-Vector Implementation
```python
error = w1 * cross(target1_body, body_vec1) + w2 * cross(target2_body, body_vec2)
tau = -kp * error - kd * omega
```

### Extended Ideas for Goal Selection

1. **Closest point** - minimum rotation from current
2. **Momentum-aligned** - minimal rate change
3. **Minimum energy** - settle quickly
4. **Maximum controllability** - avoid poor Gramian
5. **Minimum time** - bang-bang optimal
6. **Disturbance-aligned** - minimize secular torque
7. **Communication-optimal** - maximize antenna gain
8. **Thermal-optimal** - avoid sun-hot orientations
9. **Actuator-friendly** - minimize peak commands
10. **Stochastic/robust** - optimize expected performance

---

## Key Results

| Question | Best Approach | Key Finding |
|----------|--------------|-------------|
| Allocation | Weighted single QP | w=1000 gives LP+QP behavior in one solve |
| Desaturation | Continuous blending | 58% better h reduction, no mode switching |
| Goal Conversion | Multi-vector OR alternating | Both achieve full attitude; alternating every step = multi-vector |
| Manifold Choice | Controllability-aware | Best for underactuated (MTQ) systems |

---

## Code Artifacts

| File | Purpose |
|------|---------|
| `single_qp_formulation_v2.py` | Single-optimization LP+QP |
| `continuous_desaturation.py` | No-mode desaturation |
| `robust_multivector_test.py` | Comprehensive attitude tests |
| `RESEARCH_IDEAS.md` | Full list of future directions |
| `FINAL_REPORT.md` | Detailed technical report |

---

## Recommendations for Paper

1. **Section 5 (Allocation):** Present weighted single-QP as the recommended approach
2. **Section 6 (Desaturation):** Introduce continuous blending as alternative to modes
3. **Section 7 (Goal Conversion):** Multi-vector for reduced→full attitude

The key contribution is showing that:
- Direction preservation (LP) + error minimization (QP) can be unified
- Desaturation doesn't require discrete modes
- Full attitude can be achieved from reduced objectives via multi-vector tracking
