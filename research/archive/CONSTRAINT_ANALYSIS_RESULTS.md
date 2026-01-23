# Physics Constraint Analysis: Surprising Results

## Closed-Loop Comparison (120s simulation)

| Method | |θ_final| | V̇>0 count | Notes |
|--------|----------|------------|-------|
| **QP (unconstrained)** | **2.35°** | 500 | Best performance! |
| LP | 2.94° | 479 | Good, classic approach |
| 7-Projection | 7.41° | 532 | Decent |
| 10-Combined | 9.63° | 515 | Worse than unconstrained |
| **1-Power (QPC)** | **45.02°** | 1200 | **TERRIBLE!** |
| **3-Sign** | **70.47°** | 1200 | **CATASTROPHIC!** |

## 🚨 SHOCKING FINDING: Physics Constraints HURT Performance! 🚨

The "safer" constraints (Power bound, Sign preservation) caused the system to **diverge to 45-70°** while the unconstrained QP converged to **2.35°**!

### Why Power Bound Failed

The power constraint `ω'τ ≤ max(0, ω'τ_des)` becomes problematic when:

1. When τ_des wants to **accelerate** (e.g., during converging phase where θ > 0, ω < 0, so τ_des > 0):
   - P_des = ω'τ_des < 0 (negative because ω < 0, τ_des > 0)
   - Constraint becomes: ω'τ ≤ 0 (must not inject energy)
   - But we NEED to inject energy to slow down and reverse!

2. The constraint effectively **prevents positive torque when ω < 0**, which stops the controller from doing its job.

### Why Sign Preservation Failed

When the system is converging (θ > 0, ω < 0):
- Controller wants τ > 0 to slow down the negative velocity
- Sign constraint says: "ω_i < 0 and τ_des,i > 0 → require τ_i ≥ 0"
- This sounds right... but it's **always binding** in this scenario

The problem: Sign constraint + limited actuators = can't achieve enough positive torque on weak axes → system overshoots and oscillates.

## The Fundamental Problem

**These constraints are designed for DAMPING (reducing |ω|), not REGULATION (reaching θ = 0).**

For damping:
- We always want τ opposite to ω
- Energy injection is bad
- Power bound makes sense

For regulation with PD control:
- τ_des = -K_p θ - K_d ω
- When converging (θ and ω opposite signs), we need to **inject energy** to slow down
- The constraint fights the controller

## What QPC Actually Does

Looking at the QPC code:
```python
ub_constraint = max(0, taudes_dot_omega)
```

When τ_des·ω < 0 (braking), constraint is: ω'τ ≤ 0
When τ_des·ω > 0 (accelerating), constraint is: ω'τ ≤ τ_des·ω (relaxed)

This is **better** than pure power bound, but still problematic because:
- "Accelerating" vs "braking" is defined by τ_des·ω, not by what the system needs
- During convergence, τ_des·ω can be negative even though we want to slow down

## Better Physics Constraints?

### Option A: Only constrain during explicit damping
```
If mode == "damping": ω'τ ≤ 0
Else: no power constraint
```
Requires mode switching.

### Option B: Constrain based on phase space region
```
If θ·ω > 0 (diverging): ω'τ ≤ ω'τ_des (don't accelerate more than intended)
If θ·ω < 0 (converging): no constraint (let controller work)
```

### Option C: Constrain only the perpendicular component
```
τ_perp = τ - (τ·τ̂_des)τ̂_des
||τ_perp|| ≤ k||τ_parallel||
```
Allows any magnitude in desired direction, limits deviation.

### Option D: Lyapunov with progress guarantee
```
V̇ ≤ 0 AND τ·τ̂_des ≥ α_LP
```
Ensures stability AND forward progress.

## Revised Recommendation

For **general attitude control**:
1. **Use LP or unconstrained QP** - they work!
2. **Avoid power constraints** for regulation problems
3. Power constraints only make sense for **pure damping** (detumbling)

For **detumbling/damping only**:
1. Power bound is appropriate
2. Sign preservation adds per-axis safety

For **trajectory tracking**:
1. Projection guarantee (LP+QP)
2. Feedforward preservation if available

## Key Insight

**The "physics-based" constraints we derived assume the controller wants to DAMP.**

But PD controllers do more than damp - they regulate to a setpoint. During convergence, they intentionally inject energy to slow down approach velocity.

**Constraining energy injection during convergence prevents the controller from working properly.**
