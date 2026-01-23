# Corrected Findings After Validation

**Date:** January 23, 2026

---

## Critical Corrections

### 1. Single QP Formulation: DOES NOT WORK ❌

**Previous claim:** Weighted QP can encode LP behavior
**Validation result:** 0% convergence in closed-loop, 47-53° error

**Root cause discovered:**
- The weighted objective `-w*(A@u)·τ̂ + 0.5*||A@u - τ_des||²` maximizes PROJECTION
- A torque pointing in the WRONG direction can have HIGHER projection than the correct one
- Example: tau at 65° to tau_des achieved 17× higher projection than LP's parallel solution

**Correct understanding:**
- LP with equality constraint `A@u = α·τ̂` is the RIGHT formulation
- This ensures tau is EXACTLY parallel to tau_des
- LP is already fast (~1-3ms) and exact - no need to replace it
- The two-step LP+QP with projection constraint works but requires two solves

**Recommendation:** Use LP for direction-preserving allocation. Don't try to replace it with weighted QP.

---

### 2. Continuous Desaturation: SEVERE TRADE-OFF ⚠️

**Previous claim:** 58% better momentum reduction with only 9° pointing degradation
**Validation result:** Much worse trade-off across multiple scenarios

| Config | Final h (mNm·s) | Final Error (°) |
|--------|-----------------|-----------------|
| No desat | 6.10 ± 1.77 | 15.36 |
| w=1 | 3.34 ± 0.42 | 38.66 |
| w=10 | 3.09 ± 0.36 | **50.73** |
| w=50 | 3.03 ± 0.33 | **57.14** |

**Correct understanding:**
- Continuous desaturation DOES reduce momentum (~50% reduction)
- But pointing degradation is SEVERE (3-4× worse error)
- The earlier single-scenario test was misleading

**Recommendation:** Use continuous desaturation only when momentum is critical and pointing can be relaxed. For most missions, scheduled desaturation windows are better.

---

### 3. LP+QP Beats LP in Full Orbit ✓

**Validation confirmed:** Over full orbit (5400s), LP+QP outperforms LP

| Allocator | Final Error | Mean Error |
|-----------|-------------|------------|
| LP | 7.74° | 11.06° |
| **LP+QP** | **3.39°** | 10.78° |

**Insight:** LP+QP's perpendicular component utilization helps over longer time horizons.

---

### 4. Actuator Failure: Axis-Dependent

| Scenario | Convergence Rate |
|----------|------------------|
| No failures | 100% |
| RW 0 failed | 20% |
| RW 1 failed | 80% |
| RW 0 + MTQ 1 failed | 10% |

**Insight:** Which actuator fails matters enormously. The system is not uniformly robust.

---

## Updated Recommendations

### Torque Allocation

| Scenario | Method | Rationale |
|----------|--------|-----------|
| General use | LP | Fast, exact direction preservation |
| Performance-critical | LP+QP (two-step) | Better error minimization |
| Real-time constrained | LP only | Single solve, predictable timing |

### Desaturation

| Priority | Method |
|----------|--------|
| Pointing-critical | Scheduled windows only |
| Momentum-critical | Continuous with low weight |
| Balanced | Hybrid approach |

### Multi-Vector/Alternating

- **Still valid:** Both achieve full attitude from reduced objectives
- **Alternating every timestep = multi-vector** (confirmed mathematically)
- **Adaptive switching** best for underactuated systems

---

## Nullspace Desaturation Analysis

### 4+ RW Systems (Overactuated in RW)
- **1 DOF nullspace** for 4 RWs, more for additional RWs
- Can **redistribute momentum** between wheels without affecting torque
- Useful for: balancing wheel speeds, avoiding individual saturation
- **NOT for net momentum reduction** (just moves it around)

### 3 RW + 3 MTQ Systems
- Nullspace is **entirely in MTQ directions**
- Cannot desaturate RWs via nullspace - the math doesn't support it
- **Must use explicit torque-free desaturation:**
  - MTQ produces torque τ_mtq
  - RW produces opposing torque τ_rw = -τ_mtq
  - Net torque on body = 0
  - But h_rw changes (desaturation!)
- This is NOT "free" - requires favorable B-field geometry

### Practical Desaturation Recommendation

For **3 RW + MTQ** (most CubeSats):
1. Monitor wheel momentum continuously
2. Apply **continuous torque-free desaturation** (no pointing cost!)
3. Compute τ_desat = -k_h * h projected perpendicular to B
4. MTQ produces τ_desat, RW produces -τ_desat → net body torque = 0
5. Momentum flows from RW to outside via MTQ

### Key Finding: Torque-Free Desaturation is "FREE"

Validation results (3RW+3MTQ, 500s simulation):

| Mode | Final Error | h Reduction | Notes |
|------|-------------|-------------|-------|
| Pointing only | 0.00° | -0.2% | Baseline |
| **Torque-free continuous** | **0.00°** | **33.2%** | No pointing cost! |
| Slew-only | 0.00° | 5.2% | Less effective |

The earlier finding of "severe trade-off" was due to an incorrect formulation
that ADDED desaturation torque rather than using torque-free cancellation.

**Correct formulation:**
```
τ_mtq = project(-k_h * h, perp_to_B)  # MTQ achievable
u_rw += A_rw^(-1) @ (-τ_mtq)          # RW cancels MTQ torque
```
Net torque on body = τ_mtq + τ_rw = 0, but h_rw changes!

---

## Lessons Learned

1. **Always validate in closed-loop:** Single-allocation tests are misleading
2. **Test across multiple scenarios:** Single test cases can be outliers  
3. **Full-orbit matters:** Short simulations miss important dynamics
4. **Actuator geometry matters:** Failure resilience is axis-dependent
5. **Don't replace what works:** LP is fast and correct; don't over-engineer
6. **Nullspace isn't magic:** Only helps redistribute, not reduce, momentum for RW-only nullspace

---

## Research Quality Assessment

| Finding | Initial Confidence | Post-Validation |
|---------|-------------------|-----------------|
| LP > QP for direction | High | **Confirmed** |
| Single QP works | Low | **REJECTED** |
| Continuous desat mild cost | Low | **REJECTED** |
| Multi-vector works | Medium | **Confirmed** |
| LP+QP helps long-term | Medium | **Confirmed** |
| Nullspace desat for 3RW+MTQ | Medium | **REJECTED** (wrong nullspace) |
