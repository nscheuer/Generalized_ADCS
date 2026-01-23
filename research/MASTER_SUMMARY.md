# Generalized ADCS Control Laws - Master Summary

**Last Updated:** January 23, 2026

---

## Research Questions

1. **Torque Allocation:** Best method to recreate commanded torque within actuator bounds
2. **Momentum Desaturation:** How to manage momentum without degrading pointing
3. **Goal Conversion:** How to achieve full attitude from reduced objectives

---

## Final Validated Answers

### Q1: Torque Allocation

**Answer: Use LP for direction preservation, optionally LP+QP for better error**

| Method | Direction Error | Magnitude | When to Use |
|--------|----------------|-----------|-------------|
| **LP** | 0° (exact) | Lower | Default choice |
| LP+QP | 0° (constrained) | Higher | Long maneuvers |
| ~~Single QP~~ | 65° (fails!) | - | Never |

**Key insight:** LP's equality constraint `τ = α·τ̂_des` preserves direction exactly.
Weighted QP formulations fail because maximizing projection ≠ preserving direction.

### Q2: Momentum Desaturation

**Answer: Use torque-free desaturation - it's FREE!**

| Mode | Pointing Cost | h Reduction | Mechanism |
|------|---------------|-------------|-----------|
| **Torque-free** | **0°** | **33%** | MTQ + RW cancel |
| Slew-only | 0° | 5% | During transients |
| ~~Error budget~~ | Large | Small | Don't use |

**Key insight:** Torque-free desaturation adds NO pointing error because:
- MTQ produces desaturation torque τ_d
- RW produces canceling torque -τ_d  
- Net torque on body = 0
- But momentum flows from RW to outside via MTQ!

**Formulation:**
```python
τ_desat = project(-k_h * h_rw, perpendicular_to_B)  # MTQ achievable
u_mtq_desat = solve(A_mtq, τ_desat)
u_rw_desat = solve(A_rw, -A_mtq @ u_mtq_desat)      # Cancels MTQ
# Add to pointing commands
u_mtq += u_mtq_desat
u_rw += u_rw_desat
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
