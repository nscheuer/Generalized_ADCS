# CMG Implementation Plan

**Tags:** `CMG`, `actuator`, `satellite-dynamics`, `phase-2`, `integration`
**Branch:** `CMGs`
**Created:** 2026-01-17
**Status:** Phase 1 Complete, Phase 2 Ready

---

## Overview

This document describes the complete implementation plan for adding Control Moment Gyroscope (CMG) support to the Generalized_ADCS framework.

---

## Phase Summary

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | CMG Actuator Class | ✅ Complete (46 tests passing) |
| Phase 2 | Satellite Dynamics Integration | 🔄 Ready to implement |
| Phase 3 | Controller Support | ⏳ Pending |
| Phase 4 | Trajectory Planner Support | ⏳ Pending |

---

## Phase 1: CMG Actuator Class ✅ COMPLETE

### Files Created
- `ADCS/satellite_hardware/actuators/cmg.py` (~750 lines)
- `testing/test_actuators/test_actuator_CMG.py` (46 tests)
- Modified: `ADCS/satellite_hardware/actuators/__init__.py` (CMG export)

### CMG Class Features
- **Geometry:** `spin_axis(δ)`, `torque_axis(δ)`, `momentum()`
- **Torque:** `torque(u, x, os, dmode)` = h_mag × δ̇ × a(δ)
- **Storage:** `storage_torque()` returns gimbal rate δ̇ = u
- **Jacobians:** All first-order derivatives implemented
- **Hessians:** All second-order derivatives implemented
- **Noise/Bias:** Supports BiasNoiseKS for realistic modeling

### Key Differences from RW
| Aspect | RW | CMG |
|--------|----|----|
| State | Scalar h (momentum) | Scalar δ (gimbal angle) |
| Momentum | `h × axis` (fixed direction) | `h₀ × s(δ)` (rotating) |
| ∂h/∂state | Zero | `h₀ × ds/dδ` (non-zero) |
| State derivative | `ḣ = u - J_wheel × axis^T × ω̇` | `δ̇ = u` (simple) |

---

## Phase 2: Satellite Dynamics Integration 🔄 READY

### State Vector Extension

**Current:** `x = [ω(3), q(4), h_RW(N_RW)]` → length = 7 + N_RW

**New:** `x = [ω(3), q(4), h_RW(N_RW), δ_CMG(N_CMG)]` → length = 7 + N_RW + N_CMG

### Files to Modify
- `ADCS/satellite_hardware/satellite/satellite.py`

### Implementation Steps

#### Step 1: `__init__()` - Add CMG Tracking (~line 90-98)

```python
# Add to imports (line 13)
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, CMG

# In __init__ after MTQ tracking:
self.cmg_actuators: List[CMG] = [a for a in actuators if isinstance(a, CMG)]
self.number_CMG = len(self.cmg_actuators)
self.cmg_inds = np.array([j for j in range(len(self.actuators)) if isinstance(self.actuators[j], CMG)])

# Update state_len (line 98):
self.state_len = 7 + self.number_RW + self.number_CMG

# Fix MTQ filter (line 91) to exclude CMGs:
self.mtq_actuators: List[MTQ] = [s for s in actuators if isinstance(s, MTQ)]
```

#### Step 2: New CMG State Methods (after line 231)

```python
def CMG_deltas_from_state(self, state: np.ndarray) -> np.ndarray:
    """Extract CMG gimbal angles from full state vector."""
    return state[7 + self.number_RW:]

def update_CMG_deltas(self, state_or_deltas: np.ndarray) -> None:
    """Update CMG gimbal angles from state vector or delta array."""
    if np.size(state_or_deltas) == self.state_len:
        deltas = self.CMG_deltas_from_state(state_or_deltas)
    else:
        deltas = state_or_deltas
    for i, j in enumerate(self.cmg_inds):
        self.actuators[j].update_delta(deltas[i])

def CMG_momentum(self) -> np.ndarray:
    """Return total CMG angular momentum vector (3,)."""
    if self.number_CMG == 0:
        return np.zeros(3)
    return np.sum([self.actuators[j].momentum() for j in self.cmg_inds], axis=0)

def CMG_dmomentum_ddelta(self) -> np.ndarray:
    """Return Jacobian of total CMG momentum w.r.t. gimbal angles (3 x N_CMG)."""
    if self.number_CMG == 0:
        return np.zeros((3, 0))
    return np.column_stack([self.actuators[j].dmomentum__ddelta() for j in self.cmg_inds])
```

#### Step 3: `dynamics_core()` - Main Dynamics (~line 373-405)

**Key change:** CMG momentum vector varies with gimbal angle δ.

```python
# Extract states
h_RW = x[7:7+self.number_RW]
delta_CMG = x[7+self.number_RW:] if self.number_CMG > 0 else np.array([])

# Update CMG internal states
if self.number_CMG > 0:
    self.update_CMG_deltas(delta_CMG)

# Gyroscopic coupling term: ω × h_total
h_coupling = w @ J  # J*ω term
if self.number_RW > 0:
    h_coupling += h_RW @ RWaxes  # RW contribution
if self.number_CMG > 0:
    h_coupling += self.CMG_momentum()  # CMG contribution

# Angular acceleration
wdot = (-np.cross(w, h_coupling) + total_torque) @ invJ_noRW

# State derivatives
if self.number_RW > 0:
    RW_hdot = storage_torques_RW - wdot @ RWaxes.T @ np.diag(RWjs)
if self.number_CMG > 0:
    CMG_delta_dot = np.array([self.actuators[j].storage_torque(u=u[j], x=x, os=os, dmode=dmode)
                              for j in self.cmg_inds])

# Concatenate: [wdot, qdot, RW_hdot (if any), CMG_delta_dot (if any)]
```

#### Step 4: `dynJacCore()` - Jacobians (~line 525-580)

**New CMG Jacobian blocks:**

```python
# d(ωdot)/d(ω): add CMG momentum coupling
dxdot__dx[0:3, 0:3] += -skewsym(H_CMG) @ invJ_noRW

# d(ωdot)/d(δ_CMG): from gyroscopic term ω × H_CMG(δ)
D_CMG = self.CMG_dmomentum_ddelta()  # (3 x N_CMG)
cmg_start = 7 + self.number_RW
cmg_end = cmg_start + self.number_CMG
dxdot__dx[cmg_start:cmg_end, 0:3] = (-skewsym(w) @ D_CMG).T @ invJ_noRW

# d(δdot)/d(u): diagonal block = 1 (gimbal rate = command)
for i, j in enumerate(self.cmg_inds):
    dxdot__du[j, cmg_start + i] = 1.0

# d(δdot)/d(x) = 0 (independent of state)
```

#### Step 5: `dynamics_Hessians()` - Second Derivatives (~line 582-866)

**New CMG Hessian blocks:**

```python
# d²(ωdot)/(d(ω) d(δ_i)): from gyroscopic coupling
# = -J_noRW^{-1} @ skew(h_mag_i * a_i(δ_i))

# d²(ωdot)/d(δ_i)²: from gyroscopic coupling
# = J_noRW^{-1} @ skew(ω) @ h_CMG_i

# d²(δdot)/d(...)² = 0 (linear dynamics)
```

### Edge Cases

| Configuration | State Length | Notes |
|---------------|--------------|-------|
| No actuators | 7 | Simple rigid body |
| RW only | 7 + N_RW | Backward compatible |
| CMG only | 7 + N_CMG | New configuration |
| RW + CMG | 7 + N_RW + N_CMG | Full coupling |

### Verification Tests

1. **Unit tests:** Create satellite with CMG, verify state vector size
2. **Dynamics test:** Propagate CMG satellite, verify gimbal angles evolve correctly
3. **Jacobian test:** Finite-difference verification of all new blocks
4. **Hessian test:** Finite-difference verification of all new blocks
5. **Conservation test:** Zero torque → constant total angular momentum
6. **Regression tests:** `pytest testing/test_satellite/` to ensure no regressions

---

## Phase 3: Controller Support ⏳ PENDING

### Files to Modify
- `ADCS/controllers/` - Add CMG control allocation
- Potential new file: `ADCS/controllers/cmg_steering.py`

### Key Considerations
- **Singularity avoidance:** CMGs have kinematic singularities when gimbal axes align
- **Steering laws:** Pseudoinverse, null-motion, singularity-robust methods
- **Momentum management:** CMG arrays can saturate, need desaturation strategies

---

## Phase 4: Trajectory Planner Support ⏳ PENDING

### Files to Modify
- `trajectory_planner/` - Update for CMG state dimension
- Cost functions may need CMG-specific terms

### Key Considerations
- **State dimension:** Planner must handle variable state sizes
- **Constraints:** Gimbal angle limits, gimbal rate limits
- **Singularity constraints:** Avoid trajectories through CMG singularities

---

## Mathematical Reference

### CMG Momentum
```
h_CMG(δ) = h_mag × s(δ)
s(δ) = cos(δ) × s₀ + sin(δ) × t
```
where `s₀` is initial spin axis, `t` is transverse axis perpendicular to gimbal axis.

### CMG Torque
```
τ_CMG = h_mag × δ̇ × a(δ)
a(δ) = ds/dδ = -sin(δ) × s₀ + cos(δ) × t
```

### Euler's Equation with CMGs
```
ω̇ = J_noRW⁻¹ × (τ_total - ω × h_total)
h_total = J×ω + A_RW^T × h_RW + Σ h_CMG_i(δ_i)
```

### Jacobian: d(ω̇)/d(δ)
```
d(ω̇)/d(δ_i) = -J_noRW⁻¹ × skew(ω) × dh_CMG_i/dδ_i
            = -J_noRW⁻¹ × skew(ω) × (h_mag × a(δ_i))
```

---

## How to Resume This Work

1. **Checkout the CMGs branch:**
   ```bash
   git checkout CMGs
   ```

2. **Verify Phase 1 tests pass:**
   ```bash
   pytest testing/test_actuators/test_actuator_CMG.py -v
   ```

3. **Continue with Phase 2** by modifying `satellite.py` as described above

4. **Search tags for this plan:**
   - File: `docs/plans/CMG_IMPLEMENTATION_PLAN.md`
   - Tags: CMG, actuator, satellite-dynamics
   - Branch: CMGs
