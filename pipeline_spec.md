# Compensation Block & Control Law Interface — Implementation Specification

## Overview

This document specifies two pipeline stages that sit between goal formulation and allocation:
1. **Control Law Interface**: Wraps any control law as a black box, handling input/output translation
2. **Compensation Block**: Adds feedforward and correction terms around the law

The pipeline flow is:
```
Goal Formulation → [Control Law Interface] → [Compensation Block] → Allocation
```

The compensation block receives information from the goal formulation block (passed *around* the control law) and adds terms to the law's output torque.

---

## Control Law Interface

### Purpose

Wrap any attitude control law from the literature so it can be called with a standardized interface. The law receives error signals and produces a torque (or actuator commands that are back-mapped to torque).

### Data Structures

#### LawInterface (declared by each control law)

```
LawInterface:
  # Attitude error format
  attitude_type: enum {'full', 'reduced'}

  # Angular velocity handling
  omega_type: enum {'omega_error', 'omega_raw', 'no_omega'}

  # For reduced-attitude laws: does the law expect the world
  # vector in body frame or world frame?
  world_vector_frame: enum {'body', 'world'}    # only used if attitude_type == 'reduced'

  # Quaternion conventions (only used if attitude_type == 'full')
  quat_convention: enum {'hamilton_scalar_first',   # q = [q0, q1, q2, q3]
                         'hamilton_scalar_last',     # q = [q1, q2, q3, q0]
                         'jpl'}                      # conjugate of Hamilton
  error_convention: enum {'goal_times_current_inv',  # q_e = q_g ⊗ q⁻¹
                          'current_inv_times_goal'}  # q_e = q⁻¹ ⊗ q_g

  # Output format
  output_type: enum {'torque', 'actuator_commands'}

  # If output_type == 'actuator_commands':
  #   The law assumes specific actuators and outputs commands for them.
  #   We need to know the assumed actuator model to back-map to torque.
  assumed_actuator_model: ActuatorModel or null

  # Compensation defaults: what does the law already handle internally?
  includes_gyroscopic: bool       # default: false
  includes_frame_rotation: bool   # default: false
  includes_disturbance_ff: bool   # default: false
  includes_damping: bool          # default: true (most laws that take omega have damping)
```

#### ActuatorModel (for laws that output actuator commands)

```
ActuatorModel:
  # Description of the actuators the law was designed for
  # Used to back-map actuator commands to equivalent torque

  type: enum {'reaction_wheels', 'magnetorquers', 'hybrid', 'custom'}

  # For reaction wheels: axis matrix A_rw where tau = A_rw @ u_rw
  rw_axes: matrix [3 × n_rw] or null

  # For magnetorquers: axis matrix A_mtq where dipole = A_mtq @ u_mtq
  # Torque is then: tau = -[B_b]_x @ A_mtq @ u_mtq
  mtq_axes: matrix [3 × n_mtq] or null

  # For custom: a function that maps commands to torque
  # tau = command_to_torque(u, state)
  command_to_torque: function or null
```

### Processing

#### Calling the Law

```
function call_control_law(law, attitude_output, omega_output, law_flags) -> tau_ref:

  # ================================================================
  # STEP 1: Prepare inputs based on law interface type
  # ================================================================

  if law_flags.attitude_type == 'full':
    # attitude_output is q_e (already convention-converted by goal formulation block)
    law_attitude_input = attitude_output  # quaternion error

  elif law_flags.attitude_type == 'reduced':
    # attitude_output is (b_hat, r_target) tuple
    law_attitude_input = attitude_output  # (body vector, target vector)

  # omega_output is already prepared by goal formulation block:
  #   - omega_error: P @ (omega - omega_ref_body)
  #   - omega_raw: (omega - omega_ref_body) without projection
  #   - no_omega: null
  law_omega_input = omega_output  # may be null

  # ================================================================
  # STEP 2: Call the law
  # ================================================================

  if law_flags.omega_type == 'no_omega':
    raw_output = law.compute(law_attitude_input)
  else:
    raw_output = law.compute(law_attitude_input, law_omega_input)

  # ================================================================
  # STEP 3: Process output
  # ================================================================

  if law_flags.output_type == 'torque':
    tau_ref = raw_output    # already a 3-vector torque
    return tau_ref

  elif law_flags.output_type == 'actuator_commands':
    # Back-map actuator commands to equivalent torque
    tau_ref = backmap_to_torque(raw_output, law_flags.assumed_actuator_model, state)
    return tau_ref
```

#### Back-Mapping Actuator Commands to Torque

```
function backmap_to_torque(u_commands, actuator_model, state) -> tau:
  # The law output actuator commands for its assumed hardware.
  # We compute what torque those commands would produce,
  # then pass that torque to the allocation layer for our actual hardware.

  match actuator_model.type:
    'reaction_wheels':
      # tau = A_rw @ u_rw
      # u_rw is wheel torque commands, A_rw is axis matrix
      tau = actuator_model.rw_axes @ u_commands
      return tau

    'magnetorquers':
      # tau = m × B = -[B_b]_x @ A_mtq @ u_mtq
      # Need current B field in body frame from state
      B_b = state.B_body
      dipole = actuator_model.mtq_axes @ u_commands
      tau = cross(dipole, B_b)
      return tau

    'hybrid':
      # Split commands into RW and MTQ portions
      n_rw = actuator_model.rw_axes.shape[1]
      u_rw = u_commands[:n_rw]
      u_mtq = u_commands[n_rw:]
      tau_rw = actuator_model.rw_axes @ u_rw
      dipole = actuator_model.mtq_axes @ u_mtq
      tau_mtq = cross(dipole, state.B_body)
      tau = tau_rw + tau_mtq
      return tau

    'custom':
      tau = actuator_model.command_to_torque(u_commands, state)
      return tau
```

#### Important Note: Extracting Intermediate Torque

Many control laws that output actuator commands have an internal step where they compute a desired torque before allocating. For example, a magnetic PD law might compute:

```
tau_desired = -kp * e_sigma - kd * omega
m_desired = cross(tau_desired, B_b) / dot(B_b, B_b)  # pseudoinverse
u_mtq = clip(m_desired, u_min, u_max)
```

If the intermediate `tau_desired` can be extracted, it is **preferable** to use that directly rather than back-mapping from `u_mtq`. The back-mapping recovers a potentially degraded torque (clipped, projected), while the intermediate value represents what the law actually wanted. The framework should support an optional `extract_desired_torque` method on the law:

```
function call_control_law_with_extraction(law, ...):
  if hasattr(law, 'extract_desired_torque'):
    tau_ref = law.extract_desired_torque(law_attitude_input, law_omega_input)
  else:
    # Fall back to standard call + back-mapping
    tau_ref = call_control_law(law, ...)
  return tau_ref
```

---

## Compensation Block

### Purpose

Add feedforward and correction terms around the control law to improve performance without modifying the law itself. Each term is independently toggleable. The compensated torque is what gets sent to the allocation stage.

### Data Structures

#### CompensationConfig

```
CompensationConfig:
  # Toggle flags (defaults set from law_flags, can be overridden)
  enable_gyroscopic: bool           # default: NOT law_flags.includes_gyroscopic
  enable_frame_rotation: bool       # default: NOT law_flags.includes_frame_rotation
  enable_disturbance_ff: bool       # default: NOT law_flags.includes_disturbance_ff
  enable_damping_injection: bool    # default: set by goal formulation (inject_damping flag)

  # Damping injection gain (used when enable_damping_injection == True)
  k_d_injection: float or vector [3]   # scalar or per-axis gains
    default: auto-computed from law gains if available, else user-specified

  # Disturbance source
  disturbance_source: enum {'model', 'estimator', 'none'}
    default: 'model'
```

#### CompensationInputs (from goal formulation block, passed around the law)

```
CompensationInputs:
  P: matrix [3×3]              # projection matrix
  omega_ref_body: vector [3]   # reference angular velocity, body frame (rad/s)
  goal_type: enum {'full', 'reduced', 'none'}
  inject_damping: bool         # from goal formulation omega flag logic
```

### Processing

```
function compensation_step(tau_ref, q, omega, env, state,
                           comp_inputs, comp_config) -> tau_comp:
  # tau_ref: control law output torque (3-vector)
  # q, omega: current state
  # env: environment state (B field, position, etc.)
  # state: spacecraft state (inertia, RW momentum, etc.)
  # comp_inputs: from goal formulation block
  # comp_config: compensation toggles and gains

  tau_comp = tau_ref   # start with law output

  # ================================================================
  # GYROSCOPIC COMPENSATION
  # ================================================================
  if comp_config.enable_gyroscopic:
    tau_gyro = compute_gyroscopic_torque(omega, state)
    tau_comp = tau_comp + tau_gyro

  # ================================================================
  # FRAME ROTATION FEEDFORWARD
  # ================================================================
  if comp_config.enable_frame_rotation:
    tau_frame = compute_frame_rotation_torque(comp_inputs.omega_ref_body,
                                              state, comp_config)
    tau_comp = tau_comp + tau_frame

  # ================================================================
  # DISTURBANCE FEEDFORWARD
  # ================================================================
  if comp_config.enable_disturbance_ff:
    tau_dist = compute_disturbance_feedforward(q, omega, env, state,
                                               comp_config.disturbance_source)
    tau_comp = tau_comp + tau_dist

  # ================================================================
  # DAMPING INJECTION
  # ================================================================
  if comp_config.enable_damping_injection:
    tau_damp = compute_damping_injection(omega, comp_inputs, comp_config)
    tau_comp = tau_comp + tau_damp

  return tau_comp
```

### Compensation Terms — Detailed Math

#### Gyroscopic Compensation

Euler's rotational equation for a rigid body with internal momentum (reaction wheels):

$$J\dot{\boldsymbol{\omega}} = -\boldsymbol{\omega} \times (J\boldsymbol{\omega} + \mathbf{h}_{\text{RW}}) + \boldsymbol{\tau}_{\text{ext}}$$

The gyroscopic term $\boldsymbol{\omega} \times (J\boldsymbol{\omega} + \mathbf{h}_{\text{RW}})$ couples the axes nonlinearly. Compensating for it makes the law see decoupled dynamics: $J\dot{\boldsymbol{\omega}} \approx \boldsymbol{\tau}_{\text{ext}}$.

```
function compute_gyroscopic_torque(omega, state) -> tau_gyro:
  # omega: body-frame angular velocity [3] (rad/s)
  # state.J: spacecraft inertia tensor [3×3] (kg·m²)
  # state.h_rw: reaction wheel angular momentum vector [3] (N·m·s, body frame)
  #   h_rw = sum over wheels of: I_wheel_i * Omega_i * a_hat_i
  #   where Omega_i is wheel speed (rad/s) and a_hat_i is wheel axis (body frame)

  h_total = state.J @ omega + state.h_rw
  tau_gyro = cross(omega, h_total)   # this is the term we ADD to cancel it

  # Note: we ADD this because we want to cancel the -omega x h term in Euler's eq.
  # Euler's eq: J * omega_dot = -omega x (J*omega + h_rw) + tau_ext
  # With compensation: tau_allocated = tau_law + omega x (J*omega + h_rw)
  # So effective dynamics: J * omega_dot = tau_law (decoupled)

  return tau_gyro
```

**Important**: If the control law already includes gyroscopic compensation internally (common in nonlinear Lyapunov-based laws), this term must be DISABLED to avoid double-counting. The `law_flags.includes_gyroscopic` flag controls this.

#### Frame Rotation Feedforward

For tracking goals, the reference frame rotates with angular velocity $\boldsymbol{\omega}_{\text{ref}}$. The torque needed to maintain this rotation (if the spacecraft were perfectly on track) is:

$$\boldsymbol{\tau}_{\text{frame}} = J\dot{\boldsymbol{\omega}}_{\text{ref}} - \boldsymbol{\omega}_{\text{ref}} \times J\boldsymbol{\omega}_{\text{ref}}$$

The first term ($J\dot{\boldsymbol{\omega}}_{\text{ref}}$) is the angular acceleration of the reference frame. The second term ($\boldsymbol{\omega}_{\text{ref}} \times J\boldsymbol{\omega}_{\text{ref}}$) is the gyroscopic coupling at the reference rate. Together, they represent the torque that would be needed to follow the reference trajectory in the absence of any attitude error.

Without this feedforward, the control law must reactively chase the moving reference, introducing a persistent lag proportional to the tracking rate.

```
function compute_frame_rotation_torque(omega_ref_body, state, comp_config) -> tau_frame:
  # omega_ref_body: reference angular velocity in body frame [3] (rad/s)
  # We need omega_ref_body_dot: time derivative of omega_ref_body

  omega_ref_dot = compute_omega_ref_dot(omega_ref_body, state, comp_config)

  # Feedforward torque
  tau_frame = state.J @ omega_ref_dot - cross(omega_ref_body, state.J @ omega_ref_body)

  return tau_frame


function compute_omega_ref_dot(omega_ref_body, state, comp_config) -> omega_ref_dot:
  # PRIORITY 1: Analytical derivative (for named goal types)
  #
  # For nadir pointing: omega_ref = (r × v) / ||r||²
  # Its time derivative can be computed analytically:
  #   omega_ref_dot = d/dt [(r × v) / ||r||²]
  #   = [(v × v + r × a) / ||r||² - 2(r·v)(r × v) / ||r||⁴]
  #   = [r × a / ||r||² - 2(r·v)(r × v) / ||r||⁴]
  #   (since v × v = 0, and a ≈ -mu*r/||r||³ for Keplerian)
  #
  # For orbit-rate tracking (circular orbit), omega_ref_dot ≈ 0
  # For eccentric orbits, this is nonzero but small
  #
  # The goal formulation block can provide this if the goal type supports it.

  if analytical_omega_ref_dot_available:
    return analytical_omega_ref_dot

  # PRIORITY 2: Finite difference from stored previous value
  #
  # omega_ref_dot ≈ (omega_ref_body(t) - omega_ref_body(t-dt)) / dt
  #
  # This requires storing the previous omega_ref_body value.
  # Note: omega_ref_body is in body frame, so it changes both because
  # the reference rate changes AND because the body rotates. For small dt
  # this is acceptable. For high accuracy, the world-frame omega_ref should
  # be finite-differenced before body-frame conversion.

  omega_ref_dot = (omega_ref_body - state.omega_ref_body_prev) / state.dt
  state.omega_ref_body_prev = omega_ref_body   # store for next step

  return omega_ref_dot
```

**Note on interaction with gyroscopic compensation**: When both gyroscopic compensation and frame rotation feedforward are enabled, there is potential for overlap. The gyroscopic term cancels $\boldsymbol{\omega} \times J\boldsymbol{\omega}$ using the actual angular velocity, while the frame feedforward includes $-\boldsymbol{\omega}_{\text{ref}} \times J\boldsymbol{\omega}_{\text{ref}}$ using the reference. Near the goal ($\boldsymbol{\omega} \approx \boldsymbol{\omega}_{\text{ref}}$), these are similar but not identical — the gyroscopic compensation handles the actual state, while the feedforward handles the reference trajectory. Both should be enabled simultaneously; they are not redundant because one uses measured $\boldsymbol{\omega}$ and the other uses $\boldsymbol{\omega}_{\text{ref}}$.

#### Disturbance Feedforward

Subtract estimated disturbance torques so the control law doesn't have to reject them reactively.

```
function compute_disturbance_feedforward(q, omega, env, state,
                                          source) -> tau_dist:
  match source:
    'model':
      # Physics-based disturbance models
      tau_gg = gravity_gradient_torque(q, env.r_eci, state.J)
      tau_aero = aerodynamic_torque(q, env.v_eci, env.rho, state)
      tau_mag = magnetic_residual_torque(env.B_body, state.residual_dipole)
      tau_srp = solar_radiation_torque(env.sun_eci, q, state)

      tau_dist_estimate = tau_gg + tau_aero + tau_mag + tau_srp

    'estimator':
      # From augmented state estimator or disturbance observer
      # The estimator provides a torque estimate that includes all sources
      tau_dist_estimate = state.estimated_disturbance_torque

    'none':
      return zeros(3)

  # Negate: we ADD the negative of the estimated disturbance
  # so allocation produces actuator commands that cancel it
  return -tau_dist_estimate
```

**Disturbance model formulas** (for reference, these are standard):

Gravity gradient:
$$\boldsymbol{\tau}_{\text{gg}} = \frac{3\mu}{\|\mathbf{r}\|^3}\hat{\mathbf{r}}_b \times J\hat{\mathbf{r}}_b$$
where $\hat{\mathbf{r}}_b = A(\bar{q})^T \hat{\mathbf{r}}_{\text{ECI}}$ is the nadir direction in body frame.

Magnetic residual dipole:
$$\boldsymbol{\tau}_{\text{mag}} = \mathbf{m}_{\text{res}} \times \mathbf{B}_b$$
where $\mathbf{m}_{\text{res}}$ is the spacecraft's residual magnetic dipole (body frame).

Aerodynamic (simplified):
$$\boldsymbol{\tau}_{\text{aero}} = \mathbf{r}_{cp} \times \left(-\frac{1}{2}\rho v^2 C_D A_{\text{ref}} \hat{\mathbf{v}}_b\right)$$
where $\mathbf{r}_{cp}$ is center-of-pressure offset from center of mass, $\hat{\mathbf{v}}_b$ is velocity direction in body frame.

Solar radiation pressure:
$$\boldsymbol{\tau}_{\text{srp}} = \mathbf{r}_{cp,\text{sun}} \times \left(-\frac{\Phi_{\text{sun}}}{c} A_{\text{sun}} \hat{\mathbf{s}}_b\right)$$
where $\hat{\mathbf{s}}_b$ is sun direction in body frame, $\Phi_{\text{sun}}/c$ is solar radiation pressure.

#### Damping Injection

For control laws that do not accept angular velocity input, inject damping on the constrained axes.

```
function compute_damping_injection(omega, comp_inputs, comp_config) -> tau_damp:
  # omega: current body-frame angular velocity [3] (rad/s)
  # comp_inputs.P: projection matrix from goal formulation
  # comp_inputs.omega_ref_body: reference angular velocity, body frame
  # comp_config.k_d_injection: damping gain (scalar or [3])

  omega_error_raw = omega - comp_inputs.omega_ref_body

  # Project to constrained axes only
  # For full attitude: P = I, damps all axes
  # For reduced attitude: P = I - b_hat @ b_hat.T, damps only the 2 constrained axes
  # For no goal: P = 0, no damping (nothing to damp toward)
  omega_error_projected = comp_inputs.P @ omega_error_raw

  # Apply damping gain
  if comp_config.k_d_injection is scalar:
    tau_damp = -comp_config.k_d_injection * omega_error_projected
  else:
    # Per-axis gains: element-wise multiply
    tau_damp = -comp_config.k_d_injection * omega_error_projected

  return tau_damp
```

**Gain selection guidance**: The damping gain $k_d$ should be selected to provide critical or slightly underdamped response. For a spacecraft with inertia $J$ and a proportional-only law with gain $k_p$:

$$k_d \approx 2\zeta\sqrt{k_p \cdot J_{\text{avg}}}$$

where $\zeta \approx 0.7$ for slightly underdamped response and $J_{\text{avg}}$ is a representative moment of inertia. The framework can auto-compute this if $k_p$ is available from the law, but the user can override.

---

## Top-Level Pipeline: Goal Formulation → Law → Compensation

```
function pipeline_step(goal_spec, goal_spec_next, q, omega, env, state,
                       law, law_flags, goal_config, comp_config,
                       persistent_state) -> tau_comp:

  # ================================================================
  # STAGE 1: GOAL FORMULATION
  # ================================================================
  goal_outputs = goal_formulation_step(
    goal_spec, goal_spec_next, q, omega, env,
    law_flags, goal_config, persistent_state)

  # goal_outputs contains:
  #   attitude_output, omega_output,
  #   P, omega_ref_body, goal_type, inject_damping

  # ================================================================
  # STAGE 2: CONTROL LAW
  # ================================================================
  tau_ref = call_control_law(
    law,
    goal_outputs.attitude_output,
    goal_outputs.omega_output,
    law_flags)

  # ================================================================
  # STAGE 3: COMPENSATION
  # ================================================================

  # Set damping injection from goal formulation recommendation
  comp_config.enable_damping_injection = goal_outputs.inject_damping

  # Override compensation defaults based on what the law already does
  if not comp_config.manually_configured:
    comp_config.enable_gyroscopic = not law_flags.includes_gyroscopic
    comp_config.enable_frame_rotation = not law_flags.includes_frame_rotation
    comp_config.enable_disturbance_ff = not law_flags.includes_disturbance_ff

  comp_inputs = CompensationInputs(
    P = goal_outputs.P,
    omega_ref_body = goal_outputs.omega_ref_body,
    goal_type = goal_outputs.goal_type,
    inject_damping = goal_outputs.inject_damping
  )

  tau_comp = compensation_step(
    tau_ref, q, omega, env, state,
    comp_inputs, comp_config)

  # ================================================================
  # OUTPUT: tau_comp goes to allocation stage
  # ================================================================
  return tau_comp
```

---

## Gain Scaling Guidance

When applying a control law designed for a different-sized spacecraft, gains should be scaled to maintain equivalent closed-loop behavior. The general principles:

```
For a PD-type law:  tau = -kp * e_attitude - kd * omega

  kp scales with inertia:  kp_new = kp_original * (J_new / J_original)
  kd scales with inertia:  kd_new = kd_original * (J_new / J_original)

  Equivalently, choose gains based on desired natural frequency and damping:
    kp = J * omega_n²
    kd = 2 * J * zeta * omega_n

  where omega_n is the desired closed-loop bandwidth (rad/s)
  and zeta is the damping ratio (typically 0.5-1.0)
```

For sliding mode controllers, the switching gain should scale with the maximum expected torque disturbance, not just inertia. For LQR, the Q and R matrices should be retuned — scaling Q with $J^{-2}$ and R with actuator authority is a reasonable starting point, but formal re-solution of the Riccati equation is preferred.

The framework does not auto-scale gains (it would need to understand the law's internal structure). Instead, it provides:
- A warning when the law's output torque is consistently near-zero or saturating the actuators (suggesting mis-scaled gains)
- The spacecraft's inertia tensor for the user to scale manually
- Natural frequency / damping ratio analysis of the closed-loop response if the law structure is declared

---

## State Tracking

The compensation block maintains minimal persistent state:

```
CompensationState:
  omega_ref_body_prev: vector [3]    # previous omega_ref for finite-differencing omega_ref_dot
  # Initialize to zeros
```

The control law interface is stateless from the pipeline's perspective (the law itself may have internal state like integral terms, but the pipeline doesn't manage that).

---

## Warnings

The pipeline should emit warnings for the following conditions:

```
# Control law interface warnings
if law_flags.output_type == 'actuator_commands' AND law_flags.assumed_actuator_model is null:
  ERROR "Law outputs actuator commands but no assumed actuator model declared"

if law_flags.output_type == 'actuator_commands':
  WARN "Law outputs actuator commands — prefer extracting intermediate desired torque if available"

# Compensation warnings
if comp_config.enable_gyroscopic AND law_flags.includes_gyroscopic:
  WARN "Gyroscopic compensation enabled but law declares internal gyroscopic handling — risk of double-counting"

if comp_config.enable_frame_rotation AND law_flags.includes_frame_rotation:
  WARN "Frame rotation feedforward enabled but law declares internal frame handling — risk of double-counting"

# Gain scaling warnings (runtime)
if ||tau_ref|| < 1e-8 consistently for N steps:
  WARN "Control law output is near-zero — gains may be too small for this spacecraft"

if allocation_fraction < 0.1 consistently (tau_ref much larger than achievable):
  WARN "Control law requesting much more torque than actuators can provide — gains may be too large"
```
