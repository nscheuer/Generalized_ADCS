# Physics-Based Constraint Options for Torque Allocation

## The Setup

Given:
- τ_des: desired torque from high-level controller
- τ = Au: achieved torque (constrained by actuator limits)
- Goal: find u that minimizes ||τ - τ_des||² subject to physics constraints

The question: **What physics-based constraints ensure good closed-loop behavior?**

---

## Constraint 1: Power/Energy Rate (QPC's approach)

**Constraint:** ω'τ ≤ max(0, ω'τ_des)

**Physics:** Power = τ·ω. If controller wants to brake (P_des < 0), don't allow positive power (acceleration).

**Pros:**
- Simple scalar constraint
- Directly controls energy injection
- Always feasible (τ = 0 works)

**Cons:**
- Doesn't distinguish between axes
- Can still misallocate between axes (all braking on z, none on x,y)
- Doesn't consider attitude error, only rate

**When it fails:** Controller wants [-10, -10, -10] μNm, QPC might give [+5, +5, -20] μNm. Power is negative (good!) but x,y axes are accelerating.

---

## Constraint 2: Global Lyapunov Stability

**Constraint:** V̇ = θ'K_p ω + ω'τ ≤ 0

**Physics:** Total energy (potential + kinetic) must not increase.

**Pros:**
- Formal stability guarantee
- Considers both attitude and rate error

**Cons:**
- Requires knowing K_p and θ (controller state)
- Can get stuck at non-zero equilibrium (V̇ ≤ 0 doesn't mean V → 0)
- Cross-coupling between θ and ω can be counterintuitive

**When it fails:** If θ'K_p ω is negative (converging), constraint allows positive ω'τ, which can slow convergence or cause overshoot.

---

## Constraint 3: Per-Axis Damping Preservation (Sign Constraint)

**Constraint:** If ω_i · τ_des,i < 0 (braking), require sign(τ_i) = sign(τ_des,i)

**Physics:** Never accelerate an axis when the controller wants to brake it.

**Pros:**
- Per-axis guarantees
- Simple sign constraints
- Intuitive: don't fight the controller

**Cons:**
- Doesn't constrain magnitude (can under-brake severely)
- Binary constraint, no smooth trade-off
- Ignores attitude error

**When it fails:** τ_des = [-100, -100, -100], τ = [-0.001, -0.001, -0.001]. Signs match but almost no braking.

---

## Constraint 4: Per-Axis Lyapunov

**Constraint:** (τ_i + K_p θ_i) ω_i ≤ 0 for each axis

**Physics:** Each axis individually must not inject energy (considering its own potential energy K_p θ_i²).

**Pros:**
- Decoupled stability per axis
- Accounts for both θ and ω on each axis

**Cons:**
- Very restrictive (axes are coupled in reality)
- Often infeasible when θ_i and ω_i have same sign
- Requires K_p and θ

**When it fails:** When θ_i > 0 and ω_i > 0, requires τ_i ≤ -K_p θ_i. If this is outside achievable set, constraint is infeasible.

---

## Constraint 5: Angular Momentum Conservation/Bounding

**Constraint:** ||J ω̇|| ≤ ||J ω̇_des|| + ε, or equivalently ||τ - τ_ext|| ≤ ||τ_des - τ_ext|| + ε

**Physics:** Don't change angular momentum faster than intended.

**Pros:**
- Rate-of-change bound
- Physically meaningful for momentum management

**Cons:**
- Doesn't care about direction of ω̇
- External torques (τ_ext) may be unknown
- Magnitude bound, not direction

**When it fails:** Allows τ in completely wrong direction as long as magnitude is bounded.

---

## Constraint 6: Work/Energy Budget

**Constraint:** ∫ τ·ω dt ≤ ∫ τ_des·ω dt (over timestep)

Simplified: τ·ω·Δt ≤ τ_des·ω·Δt, i.e., τ·ω ≤ τ_des·ω

**Physics:** Don't do more mechanical work than intended. Same as Constraint 1 but with equality-ish bound.

**Pros:**
- Energy-based
- Prevents over-acceleration

**Cons:**
- Same as power constraint
- Discretization issues at low Δt

---

## Constraint 7: Controlled Settling Time / Eigenvalue Placement

**Constraint:** The effective closed-loop system τ = f(τ_des) should preserve stability margins.

For linearized system: ẋ = (A - BK)x, where K is the controller gain.
If allocation modifies K → K', require eigenvalues of (A - BK') still in left half-plane with margin.

**Physics:** Closed-loop poles should remain stable with adequate damping ratio.

**Pros:**
- Directly ensures closed-loop stability
- Can enforce settling time / damping ratio

**Cons:**
- Requires linearized model
- Computationally expensive (eigenvalue constraint is nonconvex)
- State-dependent

**When it fails:** Works in theory but hard to implement as a real-time constraint.

---

## Constraint 8: Passivity / Positive Real Constraint

**Constraint:** The allocation should preserve passivity of the closed-loop system.

For τ = G(τ_des), require G to be passive: ∫ τ_des' τ dt ≥ 0

**Physics:** System should dissipate energy, not create it (in a passivity sense).

**Pros:**
- Strong stability guarantees (passivity → L2 stability)
- Robust to uncertainties

**Cons:**
- Hard to enforce pointwise
- May be too restrictive
- Passivity is about input-output, not state

---

## Constraint 9: Trajectory/Feedforward Preservation

**Constraint:** If τ_des = τ_ff + τ_fb, prioritize τ_ff over τ_fb.

Specifically: minimize ||τ - τ_des||² subject to τ_ff component being achieved first.

**Physics:** Feedforward follows the planned trajectory; feedback corrects errors. If we must sacrifice something, sacrifice feedback (slower convergence) not feedforward (trajectory deviation).

**Pros:**
- Maintains trajectory tracking during saturation
- Clear priority structure

**Cons:**
- Requires knowing the τ_ff / τ_fb split
- Not all controllers separate these cleanly

**Implementation:**
1. First: maximize α such that τ contains α·τ_ff
2. Then: minimize ||τ - τ_des||² subject to τ·τ̂_ff ≥ α·||τ_ff||

---

## Constraint 10: Reachable Set / Tube Constraint

**Constraint:** The achieved τ should keep the state trajectory within a "tube" around the nominal trajectory.

Given x(t) and x_des(t), require ||x(t+Δt) - x_des(t+Δt)|| ≤ tolerance.

With ẋ = f(x) + g(x)τ: x(t+Δt) ≈ x + Δt·(f(x) + g(x)τ)

**Constraint:** ||x + Δt·(f(x) + g(x)τ) - x_des(t+Δt)|| ≤ ε

**Physics:** State evolution constraint—ensure we stay on track.

**Pros:**
- Directly constrains what we care about (state)
- Can handle nonlinear dynamics

**Cons:**
- Requires forward prediction
- Nonconvex in general
- Depends on Δt and model accuracy

---

## Comparison Table

| # | Constraint | Type | Stability | Per-Axis | Requires | Convex? |
|---|------------|------|-----------|----------|----------|---------|
| 1 | Power bound | Energy | Partial | No | ω | Yes |
| 2 | Global Lyapunov | Energy | Yes | No | θ, ω, K_p | Yes |
| 3 | Sign preservation | Damping | Partial | Yes | ω, τ_des | Yes |
| 4 | Per-axis Lyapunov | Energy | Yes | Yes | θ, ω, K_p | Yes |
| 5 | Momentum rate | Momentum | No | No | τ_des | Yes |
| 6 | Work budget | Energy | Partial | No | ω, τ_des | Yes |
| 7 | Eigenvalue | Stability | Yes | No | Model | No |
| 8 | Passivity | I/O | Yes | No | History | Hard |
| 9 | FF preservation | Trajectory | Partial | No | τ_ff, τ_fb | Yes |
| 10 | Tube constraint | State | Yes | No | Model, x_des | No |

---

## Recommended Combinations

### A) Minimum Viable (Current QPC)
- Power bound (ω'τ ≤ max(0, ω'τ_des))
- Simple, always feasible, partial stability

### B) Per-Axis Safe
- Sign preservation per axis
- Power bound globally
- Ensures no axis accelerates against controller intent

### C) Full Lyapunov
- Global Lyapunov: V̇ ≤ 0
- Plus minimum projection: τ·τ̂_des ≥ α_LP (don't give up more than LP)
- Formal stability but may sacrifice tracking

### D) Trajectory-Aware
- Feedforward preservation
- Sign preservation on feedback portion
- Best for trajectory tracking with saturating actuators

### E) Robust/Conservative
- Per-axis Lyapunov (most restrictive)
- Fall back to LP if infeasible
- Maximum stability margins, minimum performance

---

## Key Insight

**No single constraint is perfect.** The right choice depends on:

1. **What failure mode are you protecting against?**
   - Instability → Lyapunov constraints
   - Overshoot → Sign/damping constraints
   - Trajectory deviation → Feedforward preservation

2. **What information is available?**
   - Only ω → Power bound (QPC)
   - Full state (θ, ω) → Lyapunov
   - Trajectory (τ_ff, τ_fb) → Feedforward preservation

3. **What's the cost of conservatism?**
   - Safety-critical → Use restrictive constraints
   - Performance-critical → Use minimal constraints

---

## My Recommendation

**For a general-purpose controller:**

```
min ||τ - τ_des||²
subject to:
  1. Actuator bounds: lb ≤ u ≤ ub
  2. Power bound: ω'τ ≤ max(0, ω'τ_des)  [don't inject unexpected energy]
  3. Sign preservation: sign(τ_i) = sign(τ_des,i) where |τ_des,i| > threshold
```

This provides:
- Energy safety (constraint 2)
- Per-axis directionality (constraint 3)
- Minimum conservatism (still allows magnitude reduction)
- Always feasible (τ = 0 satisfies both)

For trajectory tracking, add:
```
  4. Feedforward projection: τ·τ̂_ff ≥ β·||τ_ff|| for some β ∈ (0,1)
```
