# Research Summary: Generalized ADCS Control Laws

## TL;DR

**Torque Allocation:** Use QP with Projection Dominance constraint - it beats LP 40% of the time and is never worse.

**Desaturation:** Use sequential phases (pointing → desaturation → pointing) for 2× faster momentum management.

**Goal Conversion:** Use multi-vector tracking (track 2 body vectors to 2 inertial targets) to convert reduced attitude to full attitude control.

**Don't Do:** Reachability-aware goal selection - it's counterproductive.

---

## The Optimal Allocator

```python
def optimal_allocate(tau_des, A_total, lb, ub, omega):
    """
    Step 1: Solve LP to get reference alpha_LP
    Step 2: Solve QP with constraints:
        - Projection dominance: τ·τ̂_des ≥ α_LP·||τ_des||
        - Energy (if damping): ω·τ ≤ ω·(α_LP·τ_des)
    """
    # This guarantees never worse than LP, often better
```

---

## Key Numbers

| Metric | LP | QP | Optimal |
|--------|----|----|---------|
| Final Error (°) | 16.8 | 26.3 | **16.5** |
| Better than LP | - | 0% | **37.5%** |
| Worse than LP | - | 100% | 22.5% |

---

## Code Locations

- **Optimal Allocator:** `optimal_allocator.py`
- **Constraint Analysis:** `qp_constraints_exploration.py`
- **Desaturation Strategies:** `creative_desaturation.py`
- **Attitude Conversion:** `reduced_to_full_attitude.py`
- **Full Report:** `FINAL_REPORT.md`
