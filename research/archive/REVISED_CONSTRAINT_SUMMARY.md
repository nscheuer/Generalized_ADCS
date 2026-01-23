# Revised Constraint Analysis: Final Results

## Closed-Loop Performance (120s simulation)

| Method | |θ_final| | Assessment |
|--------|----------|------------|
| **QP unconstrained** | **2.35°** | **BEST** |
| **3b-Sign critical** | **2.35°** | **TIED BEST** |
| Phase-aware | 2.69° | Good |
| LP | 2.94° | Good |
| **1a-Power brake only** | **3.11°** | **Good - physics-based!** |
| Proj+Dir hybrid | 7.41° | Acceptable |
| 1c-Power directional | 12.39° | Poor |
| 2c-Lyap rate-gated | 117.58° | FAILED |
| 2a-Lyap relative | 125.53° | FAILED |

## Key Finding: Revised Constraints CAN Work!

The **"1a-Power brake only"** constraint achieves **3.11°** - nearly as good as unconstrained QP (2.35°), while providing physics-based guarantees!

### What "1a-Power brake only" does:
```
If P_des < 0 (controller wants to brake):
    Constraint: ω'τ ≤ 0 (don't accelerate)
Else:
    No power constraint (let controller work)
```

### Why it works:
- When diverging (θ·ω > 0): P_des < 0, constraint applies → prevents acceleration
- When converging: P_des could be > 0 or < 0 depending on balance
  - If P_des > 0: No constraint → controller can accelerate to fight overshoot
  - If P_des < 0: Constraint applies → prevents overshoot from getting worse

## Why Other Constraints Failed

### 2a/2c Lyapunov variants: 117-125°
The Lyapunov constraints `ω'τ ≤ ω'τ_des` are **too restrictive during convergence**.

When converging with small ω:
- P_des = ω'τ_des can be very small or slightly negative
- Constraint forces τ to have small power → insufficient control authority
- System overshoots and oscillates

### 1c-Power directional: 12.39°
The constraint `ω'τ ≥ 0` when accelerating is problematic:
- During convergence, we sometimes need **negative** power (braking) even when the net intent is to slow approach
- This constraint prevents that flexibility

## The Winning Constraints

### Tier 1: Best Performance
1. **QP unconstrained** (2.35°) - No physics guarantee but works great
2. **3b-Sign critical** (2.35°) - Only constrains significant axes, ties best!

### Tier 2: Good with Physics Guarantee  
3. **1a-Power brake only** (3.11°) - Simple, effective, physics-based
4. **Phase-aware** (2.69°) - Adapts to state, slightly more complex

### Tier 3: Conservative/Safe
5. **LP** (2.94°) - Direction preserved, always works
6. **Proj+Dir hybrid** (7.41°) - Guarantees LP projection

## Final Recommendations

### For General Use: "1a-Power brake only"
```python
P_des = omega @ tau_des
if P_des < -epsilon:
    constraints.append(omega @ tau <= 0)
# else: no power constraint
```

**Why:**
- Only 0.76° worse than unconstrained (3.11° vs 2.35°)
- Physically meaningful: "don't accelerate when braking intended"
- Simple to implement
- Always feasible (τ=0 satisfies)

### For Maximum Performance: "3b-Sign critical"
```python
tau_threshold = 0.1 * norm(tau_des)
for i in range(3):
    if abs(tau_des[i]) < tau_threshold:
        continue  # Skip weak axes
    if omega[i] > 0 and tau_des[i] < 0:
        constraints.append(tau[i] <= 0)
    elif omega[i] < 0 and tau_des[i] > 0:
        constraints.append(tau[i] >= 0)
```

**Why:**
- Ties with unconstrained QP (2.35°)
- Per-axis sign guarantee on important axes
- Ignores axes where τ_des is small (they don't matter much)

### For Formal Stability: Use with Caution
The Lyapunov constraints (2a, 2c) provide formal V̇ ≤ 0 guarantees but **hurt convergence**. Only use if:
- Stability proof is legally/contractually required
- You accept slower/worse convergence
- Combine with minimum projection guarantee

## The Key Insight

> **Constraints should be CONDITIONAL on what the controller is trying to do.**

- Braking (P_des < 0): Apply energy constraints
- Accelerating (P_des > 0): Relax energy constraints
- Near equilibrium: Minimal constraints

The original QPC constraint `ω'τ ≤ max(0, ω'τ_des)` is close but still too restrictive because it constrains the **upper bound** even when accelerating. The revised "1a" only constrains when actually braking.
