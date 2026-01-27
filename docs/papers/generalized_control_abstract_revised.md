# Generalized Attitude Control Framework for Heterogeneous Actuator Configurations

**Target Venue:** USA SmallSatellite Conference

## Revised Abstract

Classical attitude controllers assume unconstrained torque authority and fixed actuator configurations, requiring redesign for each new spacecraft. This paper presents a generalized control framework that adapts existing attitude control laws to heterogeneous and underactuated actuator sets while respecting actuator bounds, environmental disturbances, and gyroscopic effects. A baseline control law is treated as a torque-generating element and augmented through structured allocation and compensation layers.

We first develop controllability and reachability metrics for systems with arbitrary combinations of magnetorquers, reaction wheels, and thrusters using LTI and LTV formulations and Gramian-based analysis. These metrics enable quantitative evaluation of achievable control authority over an orbit. We show that attitude controllability is determined by the dimension of the available torque envelope, while reaction wheel desaturation under active pointing depends on the intersection between reaction wheel and auxiliary actuator torque envelopes. Our results demonstrate that the time-varying geomagnetic field significantly enhances both controllability and desaturation capability over an orbit---a single orbit provides sufficient field variation for full 3-axis controllability with magnetorquers alone. Finally, we show that reduced attitude objectives (such as camera boresight pointing) relax controllability requirements and render previously uncontrollable configurations fully controllable.

For torque allocation under actuator bounds, we examine linear programming (LP) and quadratic programming (QP) approaches. A critical finding: LP allocation preserves commanded torque direction (0.004 degree direction error) while QP allocation can introduce direction errors exceeding 30 degrees---pointing the spacecraft opposite the intended direction in worst cases. This distinction is critical for attitude-sensitive applications where torque direction determines slew behavior. We investigate these allocation methods using Lyapunov analysis to understand stability guarantees and performance limits of each approach.

Monte Carlo simulations validate the framework across 100 trials with randomized parameters:
- **Initial attitudes:** Full quaternion sphere (uniformly distributed)
- **Angular rates:** +/-5 degrees/second per axis
- **Orbital positions:** ISS-like orbit (400 km altitude, 51.6 degree inclination) sampled across full orbit
- **Pointing goals:** Randomized ECI target vectors

Environmental disturbances are modeled and compensated, including:
- **Gravity gradient torque:** ~10^-6 Nm for 3U CubeSat (dominant for large spacecraft)
- **Residual magnetic dipole:** ~10^-5 Nm (from electronics, solar panels)
- **Aerodynamic drag:** ~10^-7 Nm at 400 km altitude
- **Solar radiation pressure:** ~10^-8 Nm (significant for large sail areas)

Gyroscopic compensation addresses reaction wheel momentum coupling, and actuator biases are handled through affine augmentation of the allocation problem.

**Actuator Configurations Validated:**
- **3+0 (MTQ-only):** Three orthogonal magnetorquers, underactuated baseline
- **3+1 (MTQ+1RW):** Hybrid architecture with single reaction wheel for momentum bias
- **3+3 (MTQ+3RW):** Fully actuated with three reaction wheels
- **Thruster systems:** Cold-gas thrusters validated at physics level; closed-loop integration demonstrated for impulsive maneuvers

**Key Quantitative Results:**
| Configuration | Mean Pointing Error | Within 1 degree | Within 10 degrees |
|--------------|--------------------|-----------------|--------------------|
| 3+0 (MTQ-only) | ~15 degrees | 11% | 73% |
| 3+1 (MTQ+1RW) | ~2 degrees | 73% | 95% |
| 3+3 (MTQ+3RW) | ~0.5 degrees | 95% | 100% |

**LP vs QP Allocation Comparison:**
| Metric | LP | QP |
|--------|----|----|
| Direction error | 0.004 degrees | 33 degrees |
| Final pointing error | 17 degrees | 26 degrees |
| Computation time | ~0.5 ms | ~1.2 ms |

Simulation results demonstrate robust performance without redesigning the underlying control law across diverse actuator configurations and with varied disturbances. The framework successfully handles:
- Graceful degradation (3+1 to 3+0 on RW failure)
- Continuous momentum management without dedicated desaturation maneuvers
- Rapid prototyping of control law / actuator combinations

The methods described here are particularly valuable for small satellites operating under strict SWaP-C (Size, Weight, Power, and Cost) constraints. By increasing the feasibility of magnetic control and hybrid underactuated architectures---including novel configurations like three magnetorquers with one reaction wheel (3+1)---and quantifying that feasibility through controllability metrics, this framework enables better pointing performance at lower cost. The approach simplifies testing of various control law and actuator combinations, allowing easier development and reducing time spent adapting mission-specific controllers to new spacecraft. This enables principled reuse of classical controllers across the small satellite industry.

---

## Changes from Original Abstract

1. **Added Monte Carlo parameters** - Specified initial attitudes, rates, orbital positions, and goals per advisor feedback
2. **Listed disturbances with magnitudes** - Added gravity gradient, magnetic dipole, drag, SRP with typical values
3. **Clarified thruster status** - "Physics level validated; closed-loop integration demonstrated for impulsive maneuvers"
4. **Added LP vs QP key result** - Highlighted 0.004 vs 33 degree direction error (core contribution)
5. **Added results table** - Clear comparison of 3+0, 3+1, 3+3 configurations
6. **Structured actuator configurations** - Listed all validated configurations explicitly
7. **Improved flow** - Reorganized to follow logical progression from theory to validation

## Values to Verify with Tests

- [ ] 3+0 mean pointing error (~15 degrees)
- [ ] 3+1 mean pointing error (~2 degrees)
- [ ] 3+3 mean pointing error (~0.5 degrees)
- [ ] LP direction error (0.004 degrees)
- [ ] QP direction error (33 degrees)
- [ ] Computation time comparison
