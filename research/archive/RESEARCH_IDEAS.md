# Extended Research Ideas

## Question 1: Torque Allocation Methods

### Methods to Explore (Beyond LP/QP)

1. **Convex Relaxation Methods**
   - Second-Order Cone Programming (SOCP) for CMG singularities
   - Semidefinite relaxation for non-convex actuator constraints
   - Sum-of-squares (SOS) for polynomial constraints

2. **Hierarchical/Lexicographic Optimization**
   - Multi-objective QP with priority levels
   - Preemptive goal programming
   - ε-constraint method

3. **Robust Allocation**
   - Min-max formulation against B-field uncertainty
   - Chance-constrained allocation (probabilistic bounds)
   - Tube-based robust MPC

4. **Learning-Based Approaches**
   - Neural network approximation of optimal allocator
   - Reinforcement learning for adaptive allocation
   - Imitation learning from optimal solutions

5. **Explicit Solutions**
   - Parametric programming (precompute allocation maps)
   - Lookup tables with interpolation
   - Analytical solutions for special geometries

6. **Hybrid Integer Methods**
   - Mixed-integer QP for mode switching
   - Binary constraints for actuator on/off
   - Combinatorial optimization for redundant systems

7. **Decomposition Methods**
   - Dantzig-Wolfe decomposition for large systems
   - Benders decomposition
   - Alternating Direction Method of Multipliers (ADMM)

8. **Barrier/Interior Point Variants**
   - Log-barrier for smooth actuator limits
   - Central path methods
   - Self-concordant barriers

9. **Projection-Based Methods**
   - Iterative projection onto constraint sets
   - Dykstra's algorithm
   - Douglas-Rachford splitting

10. **Control Lyapunov Function (CLF) Integration**
    - CLF-QP formulations
    - Control Barrier Functions (CBF) for safety
    - CLF-CBF-QP unified framework

---

## Question 2: Momentum Desaturation

### Desaturation Methods Beyond Sequential/Scheduled

1. **Continuous Blended Desaturation**
   - Add desaturation torque as secondary objective in QP
   - Weight adaptively based on momentum state
   - No mode switching needed

2. **Nullspace Desaturation (overactuated)**
   - Project desaturation into pointing nullspace
   - Zero impact on primary objective
   - Works for 4+ RW systems

3. **Cross-Coupling Exploitation**
   - Use gyroscopic coupling naturally
   - Design maneuvers that inherently desaturate
   - "Free" desaturation during slews

4. **Predictive Desaturation (MPC)**
   - Optimize over future B-field trajectory
   - Plan desaturation windows
   - Anticipate high-activity periods

5. **Environmental Torque Harvesting**
   - Use gravity gradient for bias momentum
   - Solar pressure for secular desaturation
   - Aerodynamic torque in LEO

6. **Momentum Bias Operation**
   - Run wheels at non-zero bias
   - Use bias to cancel secular disturbances
   - Reduce desaturation frequency

7. **Adaptive Gain Scheduling**
   - Reduce pointing gains when h is high
   - Automatically prioritize desaturation
   - Smooth transition, no modes

8. **Optimal Slew Planning**
   - Plan attitude maneuvers that naturally desaturate
   - Eigen-axis rotations through favorable orientations
   - Combined pointing + desaturation optimization

9. **Passivity-Based Desaturation**
   - Energy shaping controllers
   - Port-Hamiltonian formulations
   - Inherently stable desaturation

10. **Consensus-Based Multi-Wheel**
    - Distribute momentum across wheels
    - Avoid individual saturation
    - Cooperative wheel management

### Specific Idea: Continuous Weighted Desaturation

Instead of modes, add desaturation to the allocation:

```
min ||A·u - τ_des||² + w(h) · ||u_rw - u_desat||²

where:
  w(h) = 0 if ||h|| < h_low
  w(h) = ((||h|| - h_low)/(h_high - h_low))² otherwise
  
  u_desat = -k_h · h (desired wheel commands to reduce h)
```

This smoothly blends pointing and desaturation without discrete modes.

---

## Question 3: Attitude Goal Conversion (Reduced → Full)

### Methods for Choosing Goal on the Constraint Manifold

1. **Closest Point (Baseline)**
   - Minimize rotation from current attitude
   - Simple, intuitive
   - Ignores dynamics

2. **Momentum-Aligned**
   - Choose goal that aligns with current ω
   - Minimal rate change needed
   - Good for coast phases

3. **Minimum Energy**
   - Minimize final kinetic energy
   - Better for settling
   - Requires solving BVP

4. **Maximum Controllability**
   - Choose goal where Gramian is largest
   - Avoid poorly controllable attitudes
   - B-field dependent

5. **Minimum Time**
   - Bang-bang optimal trajectory endpoint
   - Aggressive but fast
   - Requires accurate model

6. **Disturbance-Aligned**
   - Choose goal that minimizes secular torque
   - Reduces steady-state error
   - Mission-dependent

7. **Communication-Optimal**
   - Maximize antenna gain during slew
   - Multi-objective with pointing
   - Operations-driven

8. **Thermal-Optimal**
   - Avoid sun-hot orientations
   - Extend equipment life
   - Long-term consideration

9. **Actuator-Friendly**
   - Minimize peak actuator commands
   - Avoid saturation during slew
   - Extends actuator life

10. **Stochastic/Robust**
    - Optimize expected performance under uncertainty
    - Min-max against disturbances
    - Probabilistic constraints

### Methods to Achieve Full Attitude from Reduced Objectives

1. **Multi-Vector Tracking (Tested)**
   - Track 2+ body vectors simultaneously
   - Overdetermined → full attitude
   - Works well

2. **Alternating Goals (Tested)**
   - Switch between vector goals
   - Intersection is full attitude
   - Works but slower

3. **Cascaded Control**
   - Primary: vector alignment
   - Secondary: axial rate control
   - Hierarchical structure

4. **Axial Damping**
   - Add damping on unconstrained axis
   - Simple modification
   - May not converge to specific angle

5. **Virtual Constraint**
   - Add artificial second vector
   - Orthogonal to primary
   - Creates full constraint

6. **Hybrid Switching**
   - Full attitude near goal
   - Reduced attitude far from goal
   - Smooth transition region

7. **Potential Field**
   - Add repulsive potential on unwanted rotations
   - Gradient descent on combined potential
   - May have local minima

8. **Sliding Mode**
   - Define sliding surface including axial constraint
   - Robust to disturbances
   - Chattering issues

9. **Backstepping**
   - Design control recursively
   - Include axial dynamics
   - Provable stability

10. **Flatness-Based**
    - If system is differentially flat
    - Plan trajectory in flat output space
    - Exact tracking possible

---

## Question 4: Advanced/Combined Ideas

### Unified Allocation + Desaturation + Goal Selection

1. **Single MPC Framework**
   - Optimize allocation, desaturation, and goal together
   - Rolling horizon
   - Handles all constraints

2. **Reinforcement Learning Agent**
   - Learn policy for all three decisions
   - Adapts to mission phase
   - Handles complex trade-offs

3. **Game-Theoretic Formulation**
   - Pointing vs desaturation as players
   - Nash equilibrium allocation
   - Handles conflicting objectives

4. **Hierarchical Decomposition**
   - Slow: goal selection
   - Medium: desaturation planning
   - Fast: torque allocation
   - Time-scale separation

5. **Event-Triggered Control**
   - Only update when needed
   - Reduce computation
   - Maintain performance

### Novel Actuator Configurations

1. **Variable-Speed CMGs (VSCMGs)**
   - Combine RW and CMG
   - More flexibility
   - Complex allocation

2. **Magnetic Rods (Active)**
   - Controllable permeability
   - Different from MTQs
   - Novel dynamics

3. **Propellantless Propulsion**
   - Solar sails for attitude
   - Electrodynamic tethers
   - Momentum exchange devices

4. **Flexible Appendages**
   - Use flexible mode for control
   - Coupled dynamics
   - Active damping + attitude

---

## Validation Priorities

### Must Test
- [x] Multi-vector with different actuator configs
- [x] Alternating with different switching periods
- [ ] Continuous blended desaturation
- [ ] Single-QP formulation accuracy
- [ ] Stability proofs for key methods

### Should Test
- [ ] Robustness to parameter uncertainty
- [ ] Computational load on embedded systems
- [ ] Failure mode behavior
- [ ] Integration with guidance

### Nice to Have
- [ ] Hardware-in-loop validation
- [ ] Flight heritage comparison
- [ ] Formal verification
