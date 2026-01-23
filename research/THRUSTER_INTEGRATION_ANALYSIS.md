# Thruster Integration Analysis

## Overview

This document analyzes what changes are needed to fully integrate physically-accurate 
thrusters into the Generalized_ADCS codebase, including the Python simulation framework
and the C++ trajectory planner.

**Analysis Date:** January 23, 2026
**Status:** Planning document - no implementation yet

---

## Current Thruster Implementation Status

### What Exists ✅

1. **Thruster Actuator Class** (`ADCS/satellite_hardware/actuators/thruster.py`)
   - Torque computation: `τ = r × (n̂ · F_max · u)`
   - Force computation: `F = n̂ · F_max · u`
   - Mass flow rate: `ṁ = F / (Isp · g₀)`
   - Minimum impulse bit (MIB) quantization
   - Propellant tracking
   - Full Jacobian/Hessian implementations

2. **Satellite Integration**
   - Satellite can hold Thruster objects as actuators
   - `torque()` method is called correctly in dynamics

### What's Missing ❌

1. **Control Allocation** - LP/QP allocators don't include thrusters
2. **Translational Dynamics** - Force not propagated to orbit
3. **Mission Planning** - Fuel budget not in GoalList
4. **C++ Planner** - No thruster model
5. **Closed-loop Analysis** - MIB stability effects unknown

---

## Required Changes by Component

### 1. Orbital State / Orbit Propagation

**Current State:**
- `Orbital_State` tracks position, velocity, attitude
- No coupling between attitude control and orbit

**Changes Needed:**

```python
# In ADCS/orbits/orbital_state.py

class Orbital_State:
    def __init__(self, ...):
        ...
        # ADD: Accumulated ΔV from thrusters
        self.accumulated_dv = np.zeros(3)  # [m/s] in ECI frame
        
    def apply_thrust_impulse(self, F_body: np.ndarray, q: np.ndarray, 
                             mass: float, dt: float) -> None:
        """
        Apply thrust impulse to orbital state.
        
        Parameters
        ----------
        F_body : np.ndarray
            Force vector in body frame [N]
        q : np.ndarray
            Attitude quaternion (body→ECI)
        mass : float
            Current spacecraft mass [kg]
        dt : float
            Time step [s]
        """
        # Rotate force to ECI
        R = rot_mat(q)  # Body to ECI
        F_eci = R @ F_body
        
        # Compute ΔV
        dv = (F_eci / mass) * dt  # [m/s]
        
        # Update velocity
        self.V += dv * 1e-3  # Convert to km/s
        self.accumulated_dv += dv
```

**Complexity:** Medium
**Impact:** Changes orbit propagation, affects ephemeris predictions

---

### 2. Control Allocation (LP/QP)

**Current State:**
- `MTQ_w_RW_LP` and `MTQ_w_RW_QP` allocate between MTQ and RW
- Allocation matrix `A_total` combines MTQ and RW torque mappings

**Changes Needed:**

```python
# In ADCS/controller/mtq_w_rw_LP.py

class MTQ_w_RW_LP(Controller):
    def __init__(self, est_sat, ...):
        ...
        # ADD: Detect thrusters
        self.thrusters = [a for a in est_sat.actuators if isinstance(a, Thruster)]
        self.n_thr = len(self.thrusters)
        
        if self.n_thr > 0:
            # Build thruster torque mapping
            # τ_thr = Σ (r_i × n_i) * F_max_i * u_i
            self.A_thr = np.column_stack([
                thr.effective_torque_axis for thr in self.thrusters
            ])  # Shape: (3, n_thr)
            
            # Thruster limits: u ∈ [0, 1] or [-1, 1] if bidirectional
            self.u_thr_min = np.array([
                -1.0 if thr.bidirectional else 0.0 
                for thr in self.thrusters
            ])
            self.u_thr_max = np.ones(self.n_thr)
    
    def allocate_with_thrusters(self, tau_des, b_body, est_sat, 
                                 allow_thrusters=False):
        """
        Allocate torque including thrusters.
        
        Parameters
        ----------
        allow_thrusters : bool
            If True, thrusters can be used. If False, MTQ/RW only.
            Default False to preserve propellant.
        """
        if not allow_thrusters or self.n_thr == 0:
            return self.allocate_max_torque_in_direction(tau_des, b_body, est_sat)
        
        # Extended allocation matrix: [A_rw, A_mtq(B), A_thr]
        A_total = np.hstack([
            self.A_rw,
            -skewsym(b_body) @ self.A_mtq,
            self.A_thr
        ])
        
        # Extended bounds
        u_min = np.concatenate([...self.u_thr_min])
        u_max = np.concatenate([...self.u_thr_max])
        
        # Solve extended LP
        ...
```

**Key Design Decisions:**
1. **Thruster Priority:** Should thrusters be last resort (expensive) or equal?
   - Recommend: Add `thruster_cost_weight` parameter to penalize propellant use
   
2. **MIB in Allocation:** Handle quantization at allocation or execution?
   - Recommend: Allocation assumes continuous, execution handles MIB
   
3. **Propellant Constraint:** How to enforce fuel budget?
   - Recommend: Soft constraint via cost weight, hard constraint optional

**Complexity:** High
**Impact:** Core control algorithm changes

---

### 3. Satellite Dynamics

**Current State:**
- `satellite.py` calls `actuator.torque()` for each actuator
- Total torque summed for rotational dynamics
- No force accumulation for translation

**Changes Needed:**

```python
# In ADCS/satellite_hardware/satellite/satellite.py

class Satellite:
    def dynamics_core(self, x, u, orbital_state, dmode=None):
        ...
        # EXISTING: Compute torques
        torque_total = sum(act.torque(u[i], x, orbital_state, dmode) 
                          for i, act in enumerate(self.actuators))
        
        # ADD: Compute forces from thrusters
        force_total = np.zeros(3)
        for i, act in enumerate(self.actuators):
            if isinstance(act, Thruster):
                force_total += act.force(u[i], x, orbital_state, dmode)
        
        # ADD: Store for orbit propagation
        self._last_thrust_force = force_total
        
        # Rotational dynamics (unchanged)
        ...
    
    def get_thrust_force(self) -> np.ndarray:
        """Get last computed thrust force for orbit coupling."""
        return getattr(self, '_last_thrust_force', np.zeros(3))
```

**Complexity:** Low
**Impact:** Minor addition to dynamics

---

### 4. Mass Tracking

**Current State:**
- `Satellite.mass` is constant
- No fuel consumption during simulation

**Changes Needed:**

```python
# In ADCS/satellite_hardware/satellite/satellite.py

class Satellite:
    def __init__(self, mass, ..., fuel_mass=0.0):
        self.dry_mass = mass - fuel_mass
        self.fuel_mass = fuel_mass
        self.initial_fuel_mass = fuel_mass
    
    @property
    def mass(self):
        return self.dry_mass + self.fuel_mass
    
    def consume_propellant(self, dm: float) -> bool:
        """
        Consume propellant.
        
        Returns True if successful, False if insufficient fuel.
        """
        if dm > self.fuel_mass:
            warnings.warn("Insufficient propellant!")
            dm = self.fuel_mass
        
        self.fuel_mass -= dm
        return self.fuel_mass > 0
    
    def fuel_remaining_fraction(self) -> float:
        return self.fuel_mass / self.initial_fuel_mass if self.initial_fuel_mass > 0 else 1.0
```

**Complexity:** Low
**Impact:** Changes inertia slightly as fuel depletes (advanced: fuel slosh)

---

### 5. C++ Trajectory Planner

**Current State:**
- `Satellite.cpp/hpp` has MTQ and RW models
- `add_MTQ()`, `add_RW()` methods to add actuators
- Dynamics computed in `f()` and `gradf()` methods

**Changes Needed:**

```cpp
// In trajectory_planner/src/planner/Satellite.hpp

class Satellite {
public:
    // ADD: Thruster structure
    struct ThrusterData {
        arma::vec3 direction;      // Thrust direction (body frame, unit)
        arma::vec3 position;       // Position from CoM (body frame, m)
        double max_thrust;         // Maximum thrust (N)
        double isp;                // Specific impulse (s)
        double min_on_time;        // Minimum on-time (s)
        double cost;               // Cost weight for optimization
        arma::vec3 eff_torque_ax;  // Precomputed: position × direction × max_thrust
        bool bidirectional;
    };
    
    // ADD: Thruster storage
    std::vector<ThrusterData> thrusters;
    int n_thr = 0;
    
    // ADD: Thruster methods
    void add_thruster(arma::vec3 direction, arma::vec3 position, 
                      double max_thrust, double isp, double min_on_time,
                      double cost, bool bidirectional = false);
    void add_thruster_py(py::array_t<double> direction_py, 
                         py::array_t<double> position_py,
                         double max_thrust, double isp, double min_on_time,
                         double cost, bool bidirectional = false);
    void clear_thrusters();
    
    // MODIFY: Update these to include thrusters
    int control_N() const;  // Returns n_mtq + n_rw + n_thr
    arma::vec f(arma::vec x, arma::vec u, DYNAMICS_INFO_FORM info);
    arma::mat gradf_x(arma::vec x, arma::vec u, DYNAMICS_INFO_FORM info);
    arma::mat gradf_u(arma::vec x, arma::vec u, DYNAMICS_INFO_FORM info);
};
```

```cpp
// In trajectory_planner/src/planner/Satellite.cpp

void Satellite::add_thruster(arma::vec3 direction, arma::vec3 position,
                              double max_thrust, double isp, double min_on_time,
                              double cost, bool bidirectional) {
    ThrusterData thr;
    thr.direction = arma::normalise(direction);
    thr.position = position;
    thr.max_thrust = max_thrust;
    thr.isp = isp;
    thr.min_on_time = min_on_time;
    thr.cost = cost;
    thr.bidirectional = bidirectional;
    
    // Precompute effective torque axis
    thr.eff_torque_ax = arma::cross(position, thr.direction) * max_thrust;
    
    thrusters.push_back(thr);
    n_thr = thrusters.size();
}

arma::vec Satellite::f(arma::vec x, arma::vec u, DYNAMICS_INFO_FORM info) {
    // ... existing code ...
    
    // ADD: Thruster torque contribution
    arma::vec3 thr_torque = arma::zeros<arma::vec>(3);
    int thr_idx = n_mtq + n_rw;  // Thruster commands start here
    for (int i = 0; i < n_thr; i++) {
        double u_thr = u(thr_idx + i);
        // Clamp if not bidirectional
        if (!thrusters[i].bidirectional && u_thr < 0) {
            u_thr = 0;
        }
        thr_torque += thrusters[i].eff_torque_ax * u_thr;
    }
    
    total_torque += thr_torque;
    
    // ... rest of dynamics ...
}
```

**Key Considerations:**
1. **MIB in Optimization:** The planner produces continuous trajectories.
   MIB quantization should happen post-optimization or via integer constraints.
   
2. **Fuel as State:** Could add fuel mass as state variable for fuel-optimal planning.
   - Increases state dimension by 1
   - Adds `ṁ = -|F| / (Isp · g₀)` to dynamics

3. **Force Effects:** Planner currently ignores translational dynamics.
   For short horizons (~minutes), this is acceptable. For long horizons,
   need to model orbit changes.

**Complexity:** High
**Impact:** Core C++ changes, recompilation required

---

### 6. Goal/Mission Planning

**Current State:**
- `GoalList` sequences pointing goals
- No propellant budget awareness

**Changes Needed:**

```python
# In ADCS/CONOPS/goallist.py (conceptual)

class GoalList:
    def __init__(self, ..., fuel_budget_kg: float = None):
        self.fuel_budget_kg = fuel_budget_kg
        self.fuel_used_kg = 0.0
    
    def check_fuel_budget(self, proposed_fuel_use: float) -> bool:
        """Check if proposed fuel use is within budget."""
        if self.fuel_budget_kg is None:
            return True
        return (self.fuel_used_kg + proposed_fuel_use) <= self.fuel_budget_kg
    
    def plan_with_fuel_constraint(self, satellite, orbit):
        """
        Plan goal sequence considering fuel constraints.
        
        May reorder goals or reject high-fuel-cost maneuvers.
        """
        ...
```

**Complexity:** Medium
**Impact:** Mission planning changes

---

## Integration Priority Order

### Phase 1: Basic Integration (Recommended First)
1. **Satellite dynamics** - Add force accumulation
2. **Mass tracking** - Basic fuel consumption
3. **Control allocation** - Add thruster option with high cost weight

### Phase 2: C++ Planner
4. **Thruster in C++** - Add actuator type
5. **Dynamics update** - Include in f() and gradients
6. **Python bindings** - Expose to Python

### Phase 3: Full Integration
7. **Orbital coupling** - Thrust affects orbit
8. **Fuel-optimal planning** - Add fuel as state
9. **Mission planning** - Fuel budgets

---

## Test Plan

### Unit Tests
- [ ] Thruster torque matches `r × F`
- [ ] Force is along thrust direction
- [ ] Mass flow follows `F / (Isp · g₀)`
- [ ] MIB quantization works correctly
- [ ] Propellant tracking accumulates

### Integration Tests
- [ ] Satellite with thrusters simulates correctly
- [ ] Allocation includes thrusters when allowed
- [ ] Fuel depletes during thruster use
- [ ] Warning when fuel exhausted

### System Tests
- [ ] Pointing maneuver with thruster assist
- [ ] Fuel-limited mission sequence
- [ ] Comparison: with/without thrusters

---

## Open Questions for Discussion

1. **When should thrusters be used?**
   - Only for large slews?
   - Only when MTQ authority insufficient?
   - Always available but costly?

2. **How to handle MIB in optimization?**
   - Post-processing quantization?
   - Integer programming (slow)?
   - Smooth approximation?

3. **Orbit coupling importance?**
   - For CubeSats with tiny thrusters: negligible
   - For larger sats with N-class thrusters: significant

4. **Fuel slosh modeling?**
   - Probably overkill for this framework
   - Could add as disturbance torque if needed

---

## References

1. Wie, B. (2008). Space Vehicle Dynamics and Control (2nd ed.). AIAA.
2. Sutton & Biblarz (2016). Rocket Propulsion Elements (9th ed.). Wiley.
3. Wertz & Larson (1999). Space Mission Analysis and Design (3rd ed.).
