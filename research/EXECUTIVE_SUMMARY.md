# Generalized ADCS Control Laws - Executive Summary

**Date:** January 23, 2026  
**Status:** Research complete with validation

---

## Key Questions Investigated

1. **Torque Allocation:** Best method to recreate commanded torque within actuator bounds
2. **Momentum Desaturation:** How to add desaturation to allocation methods
3. **Goal Conversion:** How to convert between full and reduced attitude objectives

---

## Final Validated Recommendations

### 1. Torque Allocation

| Method | Use When | Performance |
|--------|----------|-------------|
| **LP** | General use, real-time | Exact direction, fast |
| **LP+QP** | Performance-critical, long maneuvers | 2× better final error |
| ~~Single QP~~ | ~~Never~~ | **FAILED** in validation |

**Key finding:** Direction preservation is critical for Lyapunov-based controllers. LP's equality constraint `τ = α·τ̂_des` guarantees this. Weighted QP approximations fail catastrophically (65° direction errors).

### 2. Momentum Desaturation  

| Approach | Pointing Impact | Momentum Reduction | Recommended For |
|----------|----------------|-------------------|-----------------|
| **Scheduled windows** | None during pointing | Periodic bursts | Most missions |
| Continuous blending | **3-4× worse error** | ~50% | Emergency only |
| Nullspace (4+ RW) | None | Redistribution only | Wheel balancing |

**Key finding:** For 3RW+MTQ systems, there is NO free lunch. Continuous desaturation severely degrades pointing. Use explicit torque-free desaturation during scheduled windows.

### 3. Goal Conversion (Reduced → Full Attitude)

| Method | Works? | Notes |
|--------|--------|-------|
| **Multi-vector** | ✓ | Track 2 body vectors to 2 inertial targets |
| **Alternating** | ✓ | Switch vectors; every-timestep = multi-vector |
| **Adaptive** | ✓ | Track larger error; best for underactuated |
| Single vector | ✗ | Cannot control axial rotation |

**Key finding:** Two non-parallel body vectors tracking two non-parallel inertial targets creates 4 constraints for 3 DOF → full attitude control.

---

## Research Artifacts

| File | Purpose |
|------|---------|
| `comprehensive_validation.py` | Main validation suite |
| `CORRECTED_FINDINGS.md` | Detailed corrections |
| `GAP_ANALYSIS.md` | Research gaps identified |
| `RESEARCH_IDEAS.md` | Future directions (50+ ideas) |

---

## What We Learned That Surprised Us

1. **Weighted QP fails spectacularly** - maximizing projection ≠ preserving direction
2. **Continuous desaturation costs more than expected** - 3-4× pointing degradation
3. **LP+QP helps over long horizons** - perpendicular components add up
4. **Actuator failures are axis-dependent** - 20% vs 80% convergence rate
5. **Alternating every timestep works** - inertia low-pass filters to multi-vector

---

## Remaining Open Questions

1. Formal Lyapunov stability proof for LP+QP allocation
2. Optimal desaturation scheduling algorithm
3. Adaptive allocation that switches LP/LP+QP based on state
4. Hardware validation of findings

---

## For the Paper

**Strongly recommend including:**
- LP vs QP comparison with direction error analysis
- LP+QP formulation with projection dominance constraint
- Multi-vector attitude conversion theorem
- Torque-free desaturation geometry analysis

**Recommend NOT including:**
- Single QP weighted formulation (doesn't work)
- Continuous desaturation (trade-off too severe for most missions)
- Nullspace desaturation for 3RW+MTQ (wrong nullspace structure)
