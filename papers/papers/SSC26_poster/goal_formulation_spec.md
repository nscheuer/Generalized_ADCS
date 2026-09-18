# Goal Formulation Block — Implementation Specification

## Overview

This document specifies the Goal Formulation Block for a generalized ADCS control pipeline. The block translates arbitrary user-specified goals into standardized error signals for any control law, handling conversions between full-attitude and reduced-attitude representations, computing reference angular velocities for tracking goals, and projecting error signals to respect unconstrained degrees of freedom.

The block sits at the front of the pipeline. Downstream blocks (control law, compensation, allocation) receive its outputs and should not need to know the original goal format.

## Architecture

### Inputs (each timestep)

| Input | Type | Source |
|-------|------|--------|
| `goal_spec` | GoalSpec object | User/scheduler |
| `goal_spec_next` | GoalSpec object (nullable) | User/scheduler (for finite differencing) |
| `q` | unit quaternion [4] | Estimator (body-to-inertial, Hamilton, scalar-first) |
| `omega` | vector [3] (rad/s) | Estimator (body frame) |
| `env` | EnvironmentState | Orbital propagator |
| `law_flags` | LawInterface | Control law declaration |
| `config` | GoalConfig | User configuration |

### Outputs (each timestep)

| Output | Type | Destination |
|--------|------|-------------|
| `attitude_output` | varies by law type | Control law |
| `omega_output` | vector [3] or null | Control law |
| `P` | matrix [3×3] | Compensation block (passed around law) |
| `omega_ref_body` | vector [3] (rad/s) | Compensation block |
| `goal_type` | enum | Compensation block |

---

## Data Structures

### GoalSpec

```
GoalSpec:
  goal_type: enum {'full', 'reduced', 'none'}

  # For full-attitude goals (one of these is populated):
  q_goal: quaternion [4]          # Hamilton, scalar-first
  dcm_goal: matrix [3×3]
  euler_goal: vector [3] + sequence string (e.g. '321')
  mrp_goal: vector [3]

  # For reduced-attitude goals:
  b_hat: unit vector [3]          # body frame direction to align
  u_spec: one of:
    - named: enum {'nadir', 'zenith', 'ram', 'anti_ram',
                   'normal', 'anti_normal', 'sun', 'anti_sun',
                   'bfield', 'anti_bfield', 'perp_bfield'}
    - vector: unit vector [3] in world frame
    - coordinate: {lat, lon, alt} or {x, y, z, frame}

  # Angular velocity (optional, for any goal type):
  omega_ref_explicit: vector [3] or null  # world frame, if user specifies directly

  # Time-varying body vector (optional):
  b_hat_next: unit vector [3] or null     # for finite differencing of b_hat
```

### LawInterface

```
LawInterface:
  attitude_type: enum {'full', 'reduced'}
  omega_type: enum {'omega_error', 'omega_raw', 'no_omega'}
  world_vector_frame: enum {'body', 'world'}     # only used if attitude_type == 'reduced'
  quat_convention: enum {'hamilton_scalar_first',  # q = [q0, q1, q2, q3]
                         'hamilton_scalar_last',   # q = [q1, q2, q3, q0]
                         'jpl'}                    # conjugate of Hamilton
  error_convention: enum {'goal_times_current_inv',  # q_e = q_g ⊗ q⁻¹
                          'current_inv_times_goal'}  # q_e = q⁻¹ ⊗ q_g
```

### GoalConfig

```
GoalConfig:
  quat_set_method: enum {'nearest', 'lyapunov', 'constraint_aware'}
    default: 'nearest'
  alternating_switch: enum {'every_step', 'threshold', 'time_based'}
    default: 'every_step'
  alternating_threshold: float (rad)
    default: 0.01  # ~0.57 deg, for threshold-based switching
  alternating_period: int (timesteps)
    default: 10  # for time-based switching
  alternating_body_vectors: (vector [3], vector [3]) or null
    default: null  # auto-select orthogonal pair
  epsilon_reg: float
    default: 1e-6  # anti-parallel regularization strength
  dt: float (seconds)
    default: 1.0  # control timestep for finite differencing
```

### EnvironmentState

```
EnvironmentState:
  r_eci: vector [3]       # spacecraft position, ECI (m)
  v_eci: vector [3]       # spacecraft velocity, ECI (m/s)
  sun_eci: unit vector [3] # sun direction, ECI
  b_eci: vector [3]       # magnetic field, ECI (T)
  time: float             # current time (s, epoch-referenced)
```

---

## Processing Pipeline

### STEP 1: Normalize Goal

Convert any goal format to the internal canonical representation.

#### 1a: Full-Attitude Goals → Canonical Quaternion

```
function normalize_full_goal(goal_spec) -> q_g:
  if goal_spec.q_goal is not null:
    q_g = goal_spec.q_goal
  elif goal_spec.dcm_goal is not null:
    q_g = dcm_to_quat(goal_spec.dcm_goal)   # standard conversion
  elif goal_spec.euler_goal is not null:
    q_g = euler_to_quat(goal_spec.euler_goal, goal_spec.euler_sequence)
  elif goal_spec.mrp_goal is not null:
    q_g = mrp_to_quat(goal_spec.mrp_goal)

  q_g = normalize(q_g)

  # Enforce scalar-positive convention to avoid discontinuities
  if q_g[0] < 0:
    q_g = -q_g

  return q_g
```

#### 1b: Reduced-Attitude Goals → Canonical (b_hat, u_hat) Pair

```
function normalize_reduced_goal(goal_spec, env) -> (b_hat, u_hat):
  b_hat = normalize(goal_spec.b_hat)
  WARN_IF ||goal_spec.b_hat|| far from 1.0 before normalization

  u_hat = resolve_world_vector(goal_spec.u_spec, env)
  u_hat = normalize(u_hat)
  WARN_IF ||u_hat_raw|| far from 1.0 before normalization (for user-specified vectors)

  return (b_hat, u_hat)


function resolve_world_vector(u_spec, env) -> vector [3]:
  if u_spec is named:
    match u_spec.name:
      'nadir':      return -env.r_eci / ||env.r_eci||
      'zenith':     return  env.r_eci / ||env.r_eci||
      'ram':        return  env.v_eci / ||env.v_eci||
      'anti_ram':   return -env.v_eci / ||env.v_eci||
      'normal':     return  (env.r_eci × env.v_eci) / ||env.r_eci × env.v_eci||
      'anti_normal':return -(env.r_eci × env.v_eci) / ||env.r_eci × env.v_eci||
      'sun':        return  env.sun_eci
      'anti_sun':   return -env.sun_eci
      'bfield':     return  env.b_eci / ||env.b_eci||
      'anti_bfield':return -env.b_eci / ||env.b_eci||
      'perp_bfield':return compute_perp_bfield(env)  # project out B component
  elif u_spec is vector:
    return u_spec.vector  # already in world frame
  elif u_spec is coordinate:
    p_target = coordinate_to_eci(u_spec, env.time)  # handles LLA, ECEF, ECI
    direction = p_target - env.r_eci
    return direction / ||direction||
```

#### 1c: Set Goal Type and Projection Matrix

```
function compute_projection(goal_type, b_hat) -> P:
  if goal_type == 'full':
    P = I_3x3
  elif goal_type == 'reduced':
    P = I_3x3 - outer(b_hat, b_hat)   # projects out b_hat component
  elif goal_type == 'none':
    P = zeros(3, 3)
  return P
```

**Note:** `P` is in body frame. `b_hat` is fixed in body frame (unless time-varying), so `P` does not change with attitude — only with goal changes.

---

### STEP 2: Compute Reference Angular Velocity (World Frame)

The reference angular velocity is what the spacecraft needs to rotate at (in world frame) to maintain or track the goal.

```
function compute_omega_ref_world(goal_spec, goal_spec_next, goal_type,
                                  q, env, config) -> omega_ref_world:

  # Priority 1: User explicitly provided omega_ref
  if goal_spec.omega_ref_explicit is not null:
    return goal_spec.omega_ref_explicit

  # Priority 2: No goal — no reference rate
  if goal_type == 'none':
    return zeros(3)

  # Priority 3: Analytical derivative (for named goal types)
  if goal_type == 'reduced' AND goal_spec.u_spec is named:
    omega_analytical = compute_analytical_omega_ref(goal_spec.u_spec.name, env)
    if omega_analytical is not null:
      # Add correction for time-varying b_hat if applicable
      if goal_spec.b_hat_next is not null:
        omega_analytical += compute_bhat_correction_analytical(
          goal_spec, q, env, config.dt)
      return omega_analytical

  # Priority 4: Finite difference
  if goal_spec_next is null:
    WARN "No next-step goal available for finite differencing; using omega_ref = 0"
    return zeros(3)

  if goal_type == 'full':
    return finite_diff_full(goal_spec, goal_spec_next, config.dt)
  elif goal_type == 'reduced':
    return finite_diff_reduced(goal_spec, goal_spec_next, q, env, config.dt)


function compute_analytical_omega_ref(name, env) -> vector [3] or null:
  # Returns omega_ref in world (ECI) frame for named goals
  # These are the minimum-norm solutions perpendicular to u_hat
  match name:
    'nadir', 'zenith':
      # orbit angular velocity: omega = (r × v) / ||r||²
      return cross(env.r_eci, env.v_eci) / dot(env.r_eci, env.r_eci)
    'ram', 'anti_ram':
      # rate of change of velocity direction (requires acceleration knowledge)
      # fall through to finite difference
      return null
    'sun', 'anti_sun':
      # sun direction changes at ~1 deg/day — near zero for most purposes
      # but finite difference handles it properly
      return null
    'normal', 'anti_normal':
      # orbit normal is nearly constant for unperturbed orbits
      return zeros(3)
    default:
      return null


function finite_diff_full(goal_spec, goal_spec_next, dt) -> vector [3]:
  # Full-attitude: quaternion finite difference
  q_g = normalize_full_goal(goal_spec)
  q_g_next = normalize_full_goal(goal_spec_next)

  # Incremental rotation from current goal to next goal
  dq = quat_multiply(q_g_next, quat_inverse(q_g))

  # Enforce short path
  if dq[0] < 0:
    dq = -dq

  # Extract axis-angle
  (angle, axis) = quat_to_axis_angle(dq)

  if abs(angle) < 1e-10:
    return zeros(3)

  # omega_ref in world frame
  return axis * angle / dt


function finite_diff_reduced(goal_spec, goal_spec_next, q, env, dt) -> vector [3]:
  # Reduced-attitude: unified formula handling both u_hat and b_hat changes
  #
  # Define c_hat = A(q) * b_hat_next  (where next b_hat currently points in world frame)
  # Then omega_ref rotates c_hat to u_hat_next
  #
  # This handles:
  #   - static b, moving u: c ≈ u(t), recovers standard tracking
  #   - moving b, static u: c ≠ u even though u doesn't change
  #   - both moving: composes naturally

  # Resolve next-step targets
  (_, u_hat_next) = normalize_reduced_goal(goal_spec_next, env_at_next_step)
  # NOTE: env_at_next_step should be propagated or approximated.
  # For 1 Hz control with orbital mechanics, linear extrapolation is sufficient:
  #   env_next.r_eci ≈ env.r_eci + env.v_eci * dt
  #   env_next.v_eci ≈ env.v_eci  (or include J2 if needed)

  b_hat_next = goal_spec.b_hat_next if goal_spec.b_hat_next is not null
               else goal_spec.b_hat   # static b_hat

  # Current body vector in world frame
  A_q = quat_to_dcm(q)               # body-to-inertial DCM
  c_hat = normalize(A_q @ b_hat_next) # where next b_hat currently points

  # Rotation from c_hat to u_hat_next
  cos_delta = clamp(dot(c_hat, u_hat_next), -1.0, 1.0)
  cross_vec = cross(c_hat, u_hat_next)
  sin_delta = norm(cross_vec)

  if sin_delta < 1e-10:
    # Nearly aligned or anti-parallel
    if cos_delta > 0:
      return zeros(3)   # already aligned, no rotation needed
    else:
      # Anti-parallel: need 180° rotation about any axis perp to c_hat
      e_perp = find_perpendicular(c_hat)
      return e_perp * pi / dt
  else:
    n_hat = cross_vec / sin_delta
    delta_theta = arccos(cos_delta)
    return n_hat * delta_theta / dt
```

---

### STEP 3: Frame Conversion

Convert reference angular velocity from world (ECI) frame to body frame.

```
function convert_omega_to_body(omega_ref_world, q) -> omega_ref_body:
  A_q = quat_to_dcm(q)        # body-to-inertial: v_inertial = A_q @ v_body
  A_q_inv = A_q.T              # inertial-to-body: v_body = A_q.T @ v_inertial
  return A_q_inv @ omega_ref_world
```

---

### STEP 4: Compute Angular Velocity Error

```
function compute_omega_error(omega, omega_ref_body, P) -> (omega_e, omega_raw_error):
  omega_raw_error = omega - omega_ref_body
  omega_e = P @ omega_raw_error
  return (omega_e, omega_raw_error)
```

---

### STEP 5: Compute Attitude Error

This is the core conversion logic. The behavior depends on the 2×2 matrix of (goal_type × law attitude_type).

#### 5a: Full Goal × Full Law

```
function attitude_full_to_full(q_g, q, law_flags) -> q_e:
  # Compute error quaternion in internal convention (goal_times_current_inv)
  q_e = quat_multiply(q_g, quat_inverse(q))

  # Enforce short rotation path
  if q_e[0] < 0:
    q_e = -q_e

  # Convert to law's convention if needed
  q_e = convert_quat_convention(q_e, 'hamilton_scalar_first', law_flags)

  return q_e
```

#### 5b: Reduced Goal × Full Law

Select a quaternion from the goal set and compute error as if full-attitude.

```
function attitude_reduced_to_full(b_hat, u_hat, q, config, law_flags) -> q_e:
  # Select goal quaternion from set using configured method
  q_g = select_from_quaternion_set(b_hat, u_hat, q, config.quat_set_method)

  # Then same as full-to-full
  q_e = quat_multiply(q_g, quat_inverse(q))
  if q_e[0] < 0:
    q_e = -q_e

  q_e = convert_quat_convention(q_e, 'hamilton_scalar_first', law_flags)
  return q_e


function select_from_quaternion_set(b_hat, u_hat, q, method) -> q_g:
  # Compute parameterization basis vectors (with anti-parallel regularization)
  (x_bar, y_bar) = compute_set_basis(b_hat, u_hat)

  if method == 'nearest':
    return select_nearest_quaternion(x_bar, y_bar, q)
  elif method == 'lyapunov':
    return select_lyapunov_quaternion(x_bar, y_bar, q, ...)  # future
  elif method == 'constraint_aware':
    return select_constraint_quaternion(x_bar, y_bar, q, ...)  # future


function compute_set_basis(v_hat, u_hat) -> (x_bar, y_bar):
  # Computes the two basis quaternions for the goal set parameterization
  # f^beta(beta) = x_bar * cos(beta) + y_bar * sin(beta)
  #
  # With anti-parallel regularization for v_hat · u_hat ≈ -1

  cos_theta = clamp(dot(v_hat, u_hat), -1.0, 1.0)
  cross_vu = cross(v_hat, u_hat)
  sin_theta = norm(cross_vu)

  # Regularized x_hat: axis perpendicular to both v and u
  e_perp = find_perpendicular(v_hat)  # arbitrary vector perp to v_hat
  x_hat = normalize(cross_vu + epsilon_reg * e_perp)

  # y_hat: perpendicular to v_hat and x_hat, guaranteed orthogonal
  y_hat = cross(v_hat, x_hat)
  # y_hat is already unit length since v_hat and x_hat are orthogonal unit vectors

  theta = arccos(cos_theta)

  # Basis quaternions (from Eq. def2 in thesis)
  x_bar = [cos(theta/2),
            x_hat[0] * sin(theta/2),
            x_hat[1] * sin(theta/2),
            x_hat[2] * sin(theta/2)]

  y_bar = [0,
            y_hat[0],
            y_hat[1],
            y_hat[2]]

  return (x_bar, y_bar)


function select_nearest_quaternion(x_bar, y_bar, q) -> q_g:
  # Find beta that minimizes geodesic distance from q to f(beta)
  # f(beta) = x_bar * cos(beta) + y_bar * sin(beta)
  #
  # Geodesic distance: d = arccos(|q · f(beta)|)
  # Minimizing d ↔ maximizing |q · f(beta)|²
  #
  # q · f(beta) = (q · x_bar) cos(beta) + (q · y_bar) sin(beta)
  #             = R cos(beta - phi)
  # where R = sqrt((q · x_bar)² + (q · y_bar)²)
  #       phi = atan2(q · y_bar, q · x_bar)
  #
  # Maximum at beta = phi (or beta = phi + pi, giving -q_g, same rotation)

  qx = quat_dot(q, x_bar)
  qy = quat_dot(q, y_bar)
  beta_optimal = atan2(qy, qx)

  q_g = scalar_multiply(x_bar, cos(beta_optimal)) +
        scalar_multiply(y_bar, sin(beta_optimal))

  return normalize(q_g)  # should already be unit, but normalize for safety
```

#### 5c: Reduced Goal × Reduced Law

```
function attitude_reduced_to_reduced(b_hat, u_hat, q, law_flags) -> attitude_output:
  if law_flags.world_vector_frame == 'body':
    # Transform world vector to body frame
    A_q = quat_to_dcm(q)
    r_body = A_q.T @ u_hat
    return (b_hat, r_body)
  elif law_flags.world_vector_frame == 'world':
    return (b_hat, u_hat)
```

#### 5d: Full Goal × Reduced Law (Alternating)

Decompose the full-attitude goal into two reduced goals whose intersection is the desired orientation. Alternate between them.

```
function attitude_full_to_reduced(q_g, q, law_flags, config, state) -> (attitude_output, P_updated, switch_info):
  # state: persistent state for alternating logic (tracks active sub-goal, step counter)

  # STEP 1: Decompose full goal into two reduced sub-goals
  (b1, u1, b2, u2) = decompose_full_to_reduced_pair(q_g, config)

  # STEP 2: Determine which sub-goal is active
  active_index = determine_active_subgoal(state, q, b1, u1, b2, u2, config)
  state.active_index = active_index
  state.step_counter += 1

  if active_index == 0:
    b_active = b1; u_active = u1
  else:
    b_active = b2; u_active = u2

  # STEP 3: Update P for active sub-goal
  P_updated = I_3x3 - outer(b_active, b_active)

  # STEP 4: Compute output
  if law_flags.world_vector_frame == 'body':
    A_q = quat_to_dcm(q)
    r_body = A_q.T @ u_active
    attitude_output = (b_active, r_body)
  elif law_flags.world_vector_frame == 'world':
    attitude_output = (b_active, u_active)

  return (attitude_output, P_updated, state)


function decompose_full_to_reduced_pair(q_g, config) -> (b1, u1, b2, u2):
  # Choose two non-parallel body vectors
  if config.alternating_body_vectors is not null:
    b1, b2 = config.alternating_body_vectors
  else:
    # Default: use two principal body axes
    # Choose the two that are most orthogonal to each other (trivially, any two principal axes)
    b1 = [1, 0, 0]  # body X
    b2 = [0, 1, 0]  # body Y

  b1 = normalize(b1)
  b2 = normalize(b2)

  WARN_IF abs(dot(b1, b2)) > 0.9  "Alternating body vectors are nearly parallel"

  # Compute the world-frame targets from the desired orientation
  A_g = quat_to_dcm(q_g)   # body-to-inertial at goal
  u1 = A_g @ b1             # where b1 should point in world frame
  u2 = A_g @ b2             # where b2 should point in world frame

  return (b1, u1, b2, u2)


function determine_active_subgoal(state, q, b1, u1, b2, u2, config) -> int:
  match config.alternating_switch:
    'every_step':
      return (state.active_index + 1) % 2

    'threshold':
      # Switch when current sub-goal error is below threshold
      A_q = quat_to_dcm(q)
      if state.active_index == 0:
        r_body = A_q.T @ u1
        error = arccos(clamp(dot(b1, r_body), -1, 1))
      else:
        r_body = A_q.T @ u2
        error = arccos(clamp(dot(b2, r_body), -1, 1))

      if error < config.alternating_threshold:
        return (state.active_index + 1) % 2
      else:
        return state.active_index

    'time_based':
      if state.step_counter % config.alternating_period == 0:
        return (state.active_index + 1) % 2
      else:
        return state.active_index
```

#### 5e: No Goal

```
function attitude_none(law_flags) -> attitude_output:
  if law_flags.attitude_type == 'full':
    # Identity quaternion error: "you're at goal"
    q_e = [1, 0, 0, 0]
    q_e = convert_quat_convention(q_e, 'hamilton_scalar_first', law_flags)
    return q_e
  elif law_flags.attitude_type == 'reduced':
    # Arbitrary aligned vectors: "you're at goal"
    b_default = [0, 0, 1]
    return (b_default, b_default)
```

---

### STEP 6: Apply Omega Flags

Determine what angular velocity signal the law actually receives.

```
function apply_omega_flags(omega_e, omega_raw_error, law_flags) -> (omega_output, inject_damping):
  match law_flags.omega_type:
    'omega_error':
      # Projected error: unconstrained axes zeroed out
      return (omega_e, false)

    'omega_raw':
      # Unprojected error: law sees all 3 components
      # We subtract omega_ref but do NOT apply P
      # This "lies" to the law — it thinks omega_raw_error is the raw angular velocity
      # For linear damping terms this achieves the correct tracking behavior
      return (omega_raw_error, false)

    'no_omega':
      # Law takes no angular velocity input
      # Signal compensation block to inject damping
      return (null, true)
```

---

### STEP 7: Convention Conversion (Thin Layer Before Law)

```
function convert_quat_convention(q_e, from_convention, law_flags) -> q_e_converted:
  # Internal convention: Hamilton, scalar-first, q_e = q_g ⊗ q⁻¹
  target_convention = law_flags.quat_convention
  target_error_convention = law_flags.error_convention

  q_out = q_e

  # Handle error direction convention
  if target_error_convention != 'goal_times_current_inv':
    # q_e_alt = q⁻¹ ⊗ q_g = conjugate(q_g ⊗ q⁻¹)  (for unit quaternions)
    q_out = quat_conjugate(q_out)

  # Handle quaternion storage convention
  match target_convention:
    'hamilton_scalar_first':
      pass  # already in this format
    'hamilton_scalar_last':
      q_out = [q_out[1], q_out[2], q_out[3], q_out[0]]
    'jpl':
      # JPL = conjugate of Hamilton
      q_out = quat_conjugate(q_out)

  return q_out
```

---

### Top-Level Function: Complete Goal Formulation

```
function goal_formulation_step(goal_spec, goal_spec_next, q, omega, env,
                                law_flags, config, persistent_state) -> outputs:

  # ================================================================
  # STEP 1: NORMALIZE GOAL
  # ================================================================
  if goal_spec.goal_type == 'full':
    q_g = normalize_full_goal(goal_spec)
    goal_type = 'full'
    b_hat = null; u_hat = null

  elif goal_spec.goal_type == 'reduced':
    (b_hat, u_hat) = normalize_reduced_goal(goal_spec, env)
    goal_type = 'reduced'
    q_g = null

  elif goal_spec.goal_type == 'none':
    goal_type = 'none'
    q_g = null; b_hat = null; u_hat = null

  P = compute_projection(goal_type,
                          b_hat if goal_type == 'reduced' else null)

  # ================================================================
  # STEP 2: COMPUTE OMEGA_REF (WORLD FRAME)
  # ================================================================
  omega_ref_world = compute_omega_ref_world(
    goal_spec, goal_spec_next, goal_type, q, env, config)

  # ================================================================
  # STEP 3: FRAME CONVERSION
  # ================================================================
  omega_ref_body = convert_omega_to_body(omega_ref_world, q)

  # ================================================================
  # STEP 4: COMPUTE OMEGA ERROR
  # ================================================================
  (omega_e, omega_raw_error) = compute_omega_error(omega, omega_ref_body, P)

  # ================================================================
  # STEP 5: COMPUTE ATTITUDE ERROR (2×2 conversion table)
  # ================================================================

  P_final = P   # may be overridden in full→reduced case

  if goal_type == 'full' AND law_flags.attitude_type == 'full':
    attitude_output = attitude_full_to_full(q_g, q, law_flags)

  elif goal_type == 'reduced' AND law_flags.attitude_type == 'full':
    attitude_output = attitude_reduced_to_full(b_hat, u_hat, q, config, law_flags)

  elif goal_type == 'reduced' AND law_flags.attitude_type == 'reduced':
    attitude_output = attitude_reduced_to_reduced(b_hat, u_hat, q, law_flags)

  elif goal_type == 'full' AND law_flags.attitude_type == 'reduced':
    (attitude_output, P_final, persistent_state) = attitude_full_to_reduced(
      q_g, q, law_flags, config, persistent_state)
    # Recompute omega_e with updated P from active sub-goal
    omega_e = P_final @ omega_raw_error

  elif goal_type == 'none':
    attitude_output = attitude_none(law_flags)

  # ================================================================
  # STEP 6: APPLY OMEGA FLAGS
  # ================================================================
  (omega_output, inject_damping) = apply_omega_flags(
    omega_e, omega_raw_error, law_flags)

  # ================================================================
  # STEP 7: CONVENTION CONVERSION
  # ================================================================
  # Already applied inside attitude_full_to_full and attitude_reduced_to_full
  # For reduced law outputs, no quaternion convention needed

  # ================================================================
  # WARNINGS
  # ================================================================
  if goal_type == 'reduced':
    theta = arccos(clamp(dot(b_hat, quat_to_dcm(q).T @ u_hat), -1, 1))
    if theta > pi - 0.01:
      WARN "Near anti-parallel: regularization active"

  if goal_type == 'full' AND law_flags.attitude_type == 'reduced':
    WARN "Full goal with reduced law: using alternating sub-goal approach"

  if goal_spec.b_hat_next is not null:
    WARN "Time-varying body vector: verify this is intentional"

  if goal_spec_next is null AND goal_type != 'none':
    if goal_spec.omega_ref_explicit is null:
      WARN "No next-step goal for finite differencing; omega_ref = 0"

  # ================================================================
  # OUTPUT
  # ================================================================
  return {
    # To control law:
    attitude_output: attitude_output,
    omega_output: omega_output,

    # To compensation block (passed around law):
    P: P_final,
    omega_ref_body: omega_ref_body,
    goal_type: goal_type,
    inject_damping: inject_damping,
  }
```

---

## Utility Functions

### find_perpendicular

```
function find_perpendicular(v) -> unit vector [3]:
  # Returns an arbitrary unit vector perpendicular to v
  # Uses the axis least aligned with v to avoid numerical issues
  abs_v = [abs(v[0]), abs(v[1]), abs(v[2])]
  if abs_v[0] <= abs_v[1] AND abs_v[0] <= abs_v[2]:
    candidate = [1, 0, 0]
  elif abs_v[1] <= abs_v[2]:
    candidate = [0, 1, 0]
  else:
    candidate = [0, 0, 1]
  return normalize(cross(v, candidate))
```

### quat_to_axis_angle

```
function quat_to_axis_angle(q) -> (angle, axis):
  # q = [q0, q1, q2, q3], scalar-first
  # Assumes unit quaternion
  sin_half = norm([q[1], q[2], q[3]])
  if sin_half < 1e-10:
    return (0.0, [1, 0, 0])  # arbitrary axis for zero rotation
  angle = 2 * atan2(sin_half, q[0])
  axis = [q[1], q[2], q[3]] / sin_half
  return (angle, axis)
```

### quat_dot

```
function quat_dot(q1, q2) -> float:
  return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
```

---

## Edge Cases and Special Handling

### Anti-Parallel Vectors (θ ≈ π)

When `dot(v_hat, u_hat) ≈ -1`, the standard parameterization basis vectors are undefined. The regularization in `compute_set_basis` handles this:

```
x_hat = normalize(cross(v, u) + epsilon_reg * e_perp)
y_hat = cross(v, x_hat)
```

This smoothly degenerates: far from anti-parallel, `epsilon_reg * e_perp` is negligible. At anti-parallel, it provides a well-defined (though arbitrary) rotation axis. The choice of `e_perp` introduces a discontinuity only at the exact anti-parallel point, which is a set of measure zero.

### Zero Rotation (θ ≈ 0)

When `v_hat ≈ u_hat`, the spacecraft is already aligned. `cross(v, u) ≈ 0` and `theta ≈ 0`. The set basis computation still works (both `x_bar` and `y_bar` are well-defined in the limit), but numerically the cross product may lose precision. This is benign: the quaternion error is near identity regardless of which goal quaternion is selected.

### Quaternion Double Cover

$\bar{q}$ and $-\bar{q}$ represent the same rotation. Throughout, we enforce `q_e[0] >= 0` (scalar-positive) to avoid sign discontinuities and ensure the control law always takes the short path. This convention must be applied consistently:
- After computing `q_e`
- After selecting from quaternion set
- After finite-differencing (enforce `dq[0] >= 0`)

### GoalList / Goal Switching

When goals switch (different goal at $t$ vs $t+\Delta t$):
- `goal_spec` and `goal_spec_next` may have different types
- If types differ, finite differencing for `omega_ref` is not meaningful
- Set `omega_ref = 0` at the transition step
- The control law handles the transient via its feedback terms

### Persistent State

The `persistent_state` object tracks:
- `active_index`: which sub-goal is active in full→reduced alternating (int, 0 or 1)
- `step_counter`: timestep counter for time-based switching (int)

Initialize: `active_index = 0`, `step_counter = 0`
Reset when goal changes.

---

## Notes for Compensation Block Integration

The compensation block receives `P`, `omega_ref_body`, `goal_type`, and `inject_damping` from the goal formulation block. It uses these as follows:

- **Damping injection** (when `inject_damping == True`): Add $-k_d P \boldsymbol{\omega}_e$ to the law's output torque. Use `P` so damping only acts on constrained axes. $k_d$ is a tunable gain.

- **Gyroscopic compensation**: $\boldsymbol{\tau}_{\text{gyro}} = \boldsymbol{\omega} \times J\boldsymbol{\omega}$. Toggle: on/off based on whether the law accounts for this internally.

- **Frame rotation compensation**: For goals defined in rotating frames (LVLH, etc.), compensate for frame angular velocity. Uses `omega_ref_body`.

- **Disturbance feedforward**: $\boldsymbol{\tau}_{\text{ff}} = -\hat{\boldsymbol{\tau}}_{\text{dist}}$ from estimator. Toggle: on/off.

Each of these is independently toggleable. The law's interface flags can specify defaults (e.g., a law that includes gyroscopic terms internally should default that toggle off).
