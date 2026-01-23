# Generalized ADCS Control Laws - Master Summary

**Last Updated:** January 23, 2026 (Afternoon Session)

---

## Research Questions

1. **Torque Allocation:** Best method to recreate commanded torque within actuator bounds
2. **Momentum Desaturation:** How to manage momentum without degrading pointing
3. **Goal Conversion:** How to achieve full attitude from reduced objectives

---

## Final Validated Answers

### Q1: Torque Allocation

**Answer: Use LP+QP with projection dominance for optimal performance**

| Method | Direction Error | Projection | Closed-Loop Performance |
|--------|----------------|------------|------------------------|
| **LP** | 0° (exact) | Baseline | 22.0° final error (underactuated) |
| **LP+QP (1°)** | ≤1° | +3% avg | 16.0° final error |
| **LP+QP (5°)** | ≤5° | +18% avg | **11.3° final error** |
| ~~Naive QP~~ | 30-60° | Higher | 72° (FAILS) |

**Key insight:** Direction preservation is essential for Lyapunov stability.
- LP preserves direction via equality constraint `τ = α·τ̂_des`
- LP+QP adds projection dominance constraint: `τ·τ̂ ≥ α_LP` AND `τ·τ̂ ≥ cos(θ_max)·|τ|`
- Naive QP maximizes projection but allows arbitrary direction → destabilizes

**Mathematical Proof:** See `LP_QP_MATHEMATICAL_PROOF.md`

### Q2: Momentum Desaturation

**Answer: Torque-free desaturation, but effectiveness depends on actuator config**

| Configuration | Mode | Pointing Cost | h Reduction | Notes |
|--------------|------|---------------|-------------|-------|
| **3MTQ+3RW** | Torque-free | **0°** | **15%** | Full cancellation possible |
| **3MTQ+1RW** | Torque-free | 6° | 9% | Partial - residual x-y torque |
| **3MTQ+1RW** | Scheduled | ~0° | 8% | Only when B ⊥ RW axis |

**Key insight for 3MTQ+3RW:** True torque-free desaturation is possible!
- MTQ produces τ_d, RW produces -τ_d → net body torque = 0
- Momentum flows from RW to outside via MTQ

**Key insight for 3MTQ+1RW (underactuated):**
- RW can only produce z-axis torque
- MTQ produces torque ⊥ to B (generally has x, y, z components)
- RW can only cancel z-component → residual x-y disturbs pointing
- **Solution:** Only desaturate when B is in x-y plane (favorable geometry)

**Formulation:**
```python
# Check if B is favorable (perpendicular to RW axis)
b_z_fraction = abs(b[2]) / np.linalg.norm(b)
if b_z_fraction < 0.3:  # B mostly in x-y plane
    τ_desat = project(-k_h * h_rw, perpendicular_to_B)
    u_mtq_desat = solve(A_mtq, τ_desat)
    u_rw_desat = solve(A_rw, -τ_desat_z)  # Only cancel z-component
```

### Q3: Goal Conversion (Reduced → Full Attitude)

**Answer: Use multi-vector tracking or alternating**

| Method | Full Attitude? | Notes |
|--------|---------------|-------|
| **Multi-vector** | ✓ 100% | Track 2 body vectors to 2 inertial targets |
| **Alternating** | ✓ 100% | Switch vectors (any frequency works) |
| **Adaptive** | ✓ | Track larger error (best for underactuated) |
| Single vector | ✗ | Cannot control axial rotation |

**Key insight:** Alternating every timestep = multi-vector in the limit
(spacecraft inertia low-pass filters the switching).

---

## Validated Test Configurations

| Config | Tested | Notes |
|--------|--------|-------|
| 3MTQ + 1RW | ✓ | Most constrained |
| 3MTQ + 3RW | ✓ | Standard CubeSat |
| 4RW Pyramid | ✓ | No MTQ |
| LEO Equatorial | ✓ | |
| LEO Polar | ✓ | |
| ISS Orbit | ✓ | |
| Full attitude | ✓ | |
| Boresight pointing | ✓ | Reduced attitude |

---

## What We Learned

### Confirmed ✓
- LP preserves direction perfectly
- LP+QP helps over long horizons
- Multi-vector achieves full attitude
- Torque-free desaturation is free
- Alternating = multi-vector mathematically

### Rejected ✗
- Single weighted QP (direction fails)
- Error-budget desaturation (unstable)
- Nullspace desat for 3RW+MTQ (wrong structure)
- Reachability-aware goal selection (counterproductive)

### Surprising Findings
- Torque-free desat has ZERO pointing cost
- Weighted QP can have 65° direction error
- Actuator failures are axis-dependent (20% vs 80%)

---

## For the Paper

**Include:**
1. LP vs QP analysis with direction preservation proof
2. LP+QP with projection dominance constraint
3. Torque-free desaturation formulation
4. Multi-vector attitude conversion theorem

**Do NOT include:**
1. Weighted single-QP (doesn't work)
2. Error-budget desaturation (poor trade-off)
3. Nullspace claims for RW+MTQ (misleading)

---

## Code Artifacts

| Purpose | Files |
|---------|-------|
| Allocation | `allocation_comparison.py`, `optimal_allocator.py` |
| Desaturation | `quick_desat_study.py`, `continuous_desaturation.py` |
| Goal Conversion | `robust_multivector_test.py`, `reduced_to_full_attitude.py` |
| Validation | `comprehensive_validation.py` |
| Reports | `CORRECTED_FINDINGS.md`, `EXECUTIVE_SUMMARY.md` |
