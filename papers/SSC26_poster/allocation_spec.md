# Torque Allocation & Momentum Management — Implementation Specification

## Overview

The allocation block is the final stage of the control pipeline. It receives a compensated desired torque $\boldsymbol{\tau}_{\text{comp}}$ and maps it to actuator commands $\mathbf{u}$ that respect hardware bounds. It also optionally incorporates momentum management objectives.

Pipeline position:
```
Goal Formulation → Control Law → Compensation → [Allocation] → Actuator Commands
```

---

## Data Structures

### ActuatorSet

```
ActuatorSet:
  # List of actuator groups. Each group has a torque mapping.
  groups: list of ActuatorGroup

  # Assembled from groups at each timestep:
  B_tau: matrix [3 × n_total]    # total torque effectiveness matrix
  u_min: vector [n_total]         # lower bounds on commands
  u_max: vector [n_total]         # upper bounds on commands
  n_total: int                    # total number of actuator commands
```

### ActuatorGroup

```
ActuatorGroup:
  type: enum {'reaction_wheel', 'magnetorquer', 'thruster', 'custom'}
  n_actuators: int
  axes: matrix [3 × n_actuators]      # actuator axis directions in body frame
  u_min: vector [n_actuators]          # command lower bounds
  u_max: vector [n_actuators]          # command upper bounds

  # For reaction wheels:
  #   Commands u are wheel torques (N·m)
  #   Torque mapping: tau = A_rw @ u  (static, body-fixed axes)
  #   A_rw = axes  (each column is a wheel axis)

  # For magnetorquers:
  #   Commands u are magnetic dipole moments (A·m²)
  #   Torque mapping: tau = -[B_b]_x @ A_mtq @ u = M_eff(t) @ u
  #   A_mtq = axes  (each column is an MTQ axis)
  #   M_eff(t) = -skew(B_b(t)) @ A_mtq  (time-varying via B field)

  # For thrusters (torque-producing pairs):
  #   Commands u are thrust levels (N) or torque commands (N·m)
  #   Torque mapping: tau = A_thr @ u  (static if body-fixed)
  #   May be time-varying for gimbaled thrusters

  # For custom (e.g., CMGs, variable-geometry):
  #   Provide a function that returns the torque effectiveness matrix
  #   given the current state
  custom_effectiveness: function(state) -> matrix [3 × n] or null
```

### AllocationConfig

```
AllocationConfig:
  method: enum {'lp', 'qp', 'qp_weighted', 'qp_constrained', 'pseudoinverse'}
    default: 'lp'

  # QP weighting matrix (for qp_weighted)
  W: matrix [3×3] or null         # torque error weighting
    default: I_3x3
  lambda_reg: float                # regularization weight on ||u||²
    default: 0.0

  # LP projection fallback
  lp_project_when_infeasible: bool   # project tau onto achievable subspace
    default: true                    # when desired direction is unattainable

  # Momentum management
  enable_desaturation: bool
    default: false
  desaturation_config: DesaturationConfig or null
```

### DesaturationConfig

```
DesaturationConfig:
  strategy: enum {'nullspace', 'weighted', 'scheduled'}

  # For nullspace desaturation (overactuated systems):
  #   Desaturate in the nullspace of the primary torque objective.
  #   No impact on attitude performance.
  h_rw_target: vector [n_rw]      # target wheel speeds (rad/s), typically 0
    default: zeros

  # For weighted desaturation:
  #   Blend desaturation objective into the QP cost.
  #   Trades pointing accuracy for momentum management.
  desat_weight: float              # relative weight of desaturation vs pointing
    default: 0.1

  # For scheduled desaturation (underactuated systems):
  #   Desaturate during periods of favorable MTQ authority.
  #   Requires knowledge of orbital B-field schedule.
  desat_authority_threshold: float  # minimum MTQ authority to attempt desaturation
    default: 0.5
```

### AllocationOutput

```
AllocationOutput:
  u: vector [n_total]              # actuator commands
  tau_achieved: vector [3]         # actual torque that will be produced
  alpha: float                     # fraction of desired torque achieved
                                   #   alpha = tau_achieved · tau_hat_desired / ||tau_desired||
  direction_error: float           # angle between desired and achieved torque (rad)
  feasible: bool                   # whether the full desired torque was achievable
```

---

## Processing Pipeline

### STEP 1: Assemble Torque Effectiveness Matrix

At each timestep, build $B_\tau(t)$ from the actuator groups.

```
function assemble_B_tau(actuator_set, state) -> (B_tau, u_min, u_max):
  # Build the full torque effectiveness matrix by horizontally stacking
  # each actuator group's contribution

  B_columns = []
  u_min_parts = []
  u_max_parts = []

  for group in actuator_set.groups:
    match group.type:

      'reaction_wheel':
        # Static mapping: tau = A_rw @ u_rw
        # Each column of A_rw is a wheel axis in body frame
        B_group = group.axes                    # [3 × n_rw]
        # Note: for CMGs or variable-speed devices, this could be
        # state-dependent. Use custom_effectiveness in that case.

      'magnetorquer':
        # Time-varying mapping: tau = -[B_b]_x @ A_mtq @ u_mtq
        B_b = state.B_body                      # magnetic field in body frame
        B_skew = skew_symmetric(B_b)            # [3×3] skew matrix
        M_eff = -B_skew @ group.axes            # [3 × n_mtq]
        B_group = M_eff

      'thruster':
        # Typically static for body-fixed thrusters
        # tau = A_thr @ u_thr
        B_group = group.axes                    # [3 × n_thr]
        # For gimbaled thrusters, use custom_effectiveness

      'custom':
        B_group = group.custom_effectiveness(state)   # [3 × n]

    B_columns.append(B_group)
    u_min_parts.append(group.u_min)
    u_max_parts.append(group.u_max)

  B_tau = horizontal_stack(B_columns)     # [3 × n_total]
  u_min = concatenate(u_min_parts)        # [n_total]
  u_max = concatenate(u_max_parts)        # [n_total]

  return (B_tau, u_min, u_max)
```

### STEP 2: Solve Allocation Problem

```
function allocate(tau_desired, B_tau, u_min, u_max, config) -> AllocationOutput:

  if norm(tau_desired) < 1e-12:
    # Zero desired torque — return zero commands
    return AllocationOutput(
      u = zeros(n_total),
      tau_achieved = zeros(3),
      alpha = 1.0,
      direction_error = 0.0,
      feasible = true
    )

  match config.method:
    'lp':
      return allocate_lp(tau_desired, B_tau, u_min, u_max, config)
    'qp':
      return allocate_qp(tau_desired, B_tau, u_min, u_max, config)
    'qp_weighted':
      return allocate_qp_weighted(tau_desired, B_tau, u_min, u_max, config)
    'qp_constrained':
      return allocate_qp_constrained(tau_desired, B_tau, u_min, u_max, config)
    'pseudoinverse':
      return allocate_pseudoinverse(tau_desired, B_tau, u_min, u_max)
```

#### LP Allocation: Direction-Preserving

Maximize torque magnitude along the desired direction.

```
function allocate_lp(tau_desired, B_tau, u_min, u_max, config) -> AllocationOutput:
  # Formulation:
  #   max   alpha
  #   s.t.  B_tau @ u = alpha * tau_hat
  #         u_min <= u <= u_max
  #         alpha >= 0
  #
  # where tau_hat = tau_desired / ||tau_desired||

  tau_hat = tau_desired / norm(tau_desired)

  # Check if desired direction is achievable at all
  # (i.e., does tau_hat lie in the column space of B_tau?)
  # For rank-deficient B_tau (e.g., MTQ-only), tau_hat may have
  # a component in the null space of B_tau^T.

  # Attempt the LP
  (u_opt, alpha_opt, status) = solve_lp(
    objective = maximize alpha,
    equality = B_tau @ u == alpha * tau_hat,
    bounds = u_min <= u <= u_max,
    alpha >= 0
  )

  if status == 'optimal' AND alpha_opt > 0:
    tau_achieved = alpha_opt * tau_hat
    return AllocationOutput(
      u = u_opt,
      tau_achieved = tau_achieved,
      alpha = alpha_opt / norm(tau_desired),
      direction_error = 0.0,  # LP preserves direction by construction
      feasible = alpha_opt >= norm(tau_desired)
    )

  elif status == 'optimal' AND alpha_opt == 0:
    # Desired direction is unachievable (e.g., along B field for MTQ-only)
    if config.lp_project_when_infeasible:
      return allocate_lp_projected(tau_desired, B_tau, u_min, u_max, config)
    else:
      return AllocationOutput(
        u = zeros(n_total),
        tau_achieved = zeros(3),
        alpha = 0.0,
        direction_error = NaN,  # undefined when zero torque
        feasible = false
      )

  else:
    # Solver failure — return zero
    WARN "LP allocation solver failed"
    return AllocationOutput(u = zeros(n_total), tau_achieved = zeros(3),
                            alpha = 0.0, direction_error = NaN, feasible = false)


function allocate_lp_projected(tau_desired, B_tau, u_min, u_max, config) -> AllocationOutput:
  # The desired torque direction is not in the achievable subspace.
  # Project tau_desired onto the achievable subspace and re-solve.
  #
  # For MTQ-only: achievable subspace is the plane perpendicular to B.
  # For general rank-deficient B_tau: project onto column space of B_tau.
  #
  # More precisely: project tau_hat onto range(B_tau).

  # Compute projection onto column space of B_tau
  # Using SVD: B_tau = U @ S @ V^T, range is spanned by columns of U with nonzero singular values
  (U, S, Vt) = svd(B_tau)
  rank = count(S > 1e-10)
  U_range = U[:, :rank]

  tau_projected = U_range @ (U_range.T @ tau_desired)

  if norm(tau_projected) < 1e-12:
    # Desired torque is entirely in the null space — nothing achievable
    return AllocationOutput(u = zeros(n_total), tau_achieved = zeros(3),
                            alpha = 0.0, direction_error = NaN, feasible = false)

  # Re-run LP with projected direction
  tau_hat_proj = tau_projected / norm(tau_projected)

  (u_opt, alpha_opt, status) = solve_lp(
    objective = maximize alpha,
    equality = B_tau @ u == alpha * tau_hat_proj,
    bounds = u_min <= u <= u_max,
    alpha >= 0
  )

  if status == 'optimal':
    tau_achieved = alpha_opt * tau_hat_proj
    direction_error = arccos(clamp(
      dot(tau_achieved, tau_desired) / (norm(tau_achieved) * norm(tau_desired)),
      -1, 1))
    return AllocationOutput(
      u = u_opt,
      tau_achieved = tau_achieved,
      alpha = dot(tau_achieved, tau_desired / norm(tau_desired)) / norm(tau_desired),
      direction_error = direction_error,
      feasible = false  # we had to project, so not fully feasible
    )
  else:
    return AllocationOutput(u = zeros(n_total), tau_achieved = zeros(3),
                            alpha = 0.0, direction_error = NaN, feasible = false)
```

#### QP Allocation: Minimum Error

Minimize the squared torque error.

```
function allocate_qp(tau_desired, B_tau, u_min, u_max, config) -> AllocationOutput:
  # Formulation:
  #   min   ||B_tau @ u - tau_desired||²_W + lambda * ||u||²
  #   s.t.  u_min <= u <= u_max
  #
  # This is a bounded least-squares problem.

  W = config.W if config.W is not null else I_3x3
  lambda_reg = config.lambda_reg

  if lambda_reg > 0:
    # Augmented system: [sqrt(W) @ B_tau; sqrt(lambda) @ I] @ u ≈ [sqrt(W) @ tau_desired; 0]
    A_aug = vertical_stack(sqrt_matrix(W) @ B_tau,
                           sqrt(lambda_reg) * I_n)
    b_aug = concatenate(sqrt_matrix(W) @ tau_desired,
                        zeros(n_total))
  else:
    A_aug = sqrt_matrix(W) @ B_tau
    b_aug = sqrt_matrix(W) @ tau_desired

  # Solve bounded least squares
  u_opt = bounded_least_squares(A_aug, b_aug, u_min, u_max)
  # e.g., scipy.optimize.lsq_linear or equivalent

  tau_achieved = B_tau @ u_opt
  direction_error = arccos(clamp(
    dot(tau_achieved, tau_desired) / (norm(tau_achieved) * norm(tau_desired) + 1e-12),
    -1, 1))
  alpha = dot(tau_achieved, tau_desired / norm(tau_desired)) / norm(tau_desired)

  return AllocationOutput(
    u = u_opt,
    tau_achieved = tau_achieved,
    alpha = alpha,
    direction_error = direction_error,
    feasible = norm(tau_achieved - tau_desired) < 1e-6
  )
```

#### QP with Direction Constraint

QP that constrains the achieved torque direction to match the desired direction, then minimizes error in magnitude.

```
function allocate_qp_constrained(tau_desired, B_tau, u_min, u_max, config) -> AllocationOutput:
  # Formulation:
  #   min   ||B_tau @ u - tau_desired||²
  #   s.t.  u_min <= u <= u_max
  #         B_tau @ u = alpha * tau_hat  for some alpha >= 0
  #         (equivalently: (I - tau_hat @ tau_hat^T) @ B_tau @ u = 0)
  #
  # This preserves direction like LP but optimizes magnitude like QP.
  # Falls back to LP if the constrained QP is infeasible.

  tau_hat = tau_desired / norm(tau_desired)

  # Direction constraint: perpendicular components are zero
  P_perp = I_3x3 - outer(tau_hat, tau_hat)
  # Constraint: P_perp @ B_tau @ u = 0

  # This is a constrained least-squares problem:
  #   min  ||B_tau @ u - tau_desired||²
  #   s.t. P_perp @ B_tau @ u = 0
  #        u_min <= u <= u_max

  (u_opt, status) = solve_constrained_qp(
    objective = norm(B_tau @ u - tau_desired)²,
    equality = P_perp @ B_tau @ u == 0,
    bounds = u_min <= u <= u_max
  )

  if status == 'optimal':
    tau_achieved = B_tau @ u_opt
    alpha = dot(tau_achieved, tau_hat) / norm(tau_desired)
    return AllocationOutput(
      u = u_opt,
      tau_achieved = tau_achieved,
      alpha = alpha,
      direction_error = 0.0,  # direction preserved by constraint
      feasible = abs(norm(tau_achieved) - norm(tau_desired)) < 1e-6
    )
  else:
    # Fall back to LP
    WARN "Constrained QP infeasible, falling back to LP"
    return allocate_lp(tau_desired, B_tau, u_min, u_max, config)
```

#### Pseudoinverse Allocation

Simple unconstrained least-squares. Useful as a baseline and for fully actuated systems where bounds are unlikely to be hit.

```
function allocate_pseudoinverse(tau_desired, B_tau, u_min, u_max) -> AllocationOutput:
  # u = B_tau^+ @ tau_desired  (Moore-Penrose pseudoinverse)
  u_opt = pinv(B_tau) @ tau_desired

  # Clip to bounds (may degrade solution)
  u_clipped = clip(u_opt, u_min, u_max)
  clipped = any(u_opt != u_clipped)
  if clipped:
    WARN "Pseudoinverse allocation required clipping — consider using QP instead"

  tau_achieved = B_tau @ u_clipped
  direction_error = arccos(clamp(
    dot(tau_achieved, tau_desired) / (norm(tau_achieved) * norm(tau_desired) + 1e-12),
    -1, 1))

  return AllocationOutput(
    u = u_clipped,
    tau_achieved = tau_achieved,
    alpha = dot(tau_achieved, tau_desired / norm(tau_desired)) / norm(tau_desired),
    direction_error = direction_error,
    feasible = not clipped
  )
```

---

### STEP 3: Momentum Management (Optional)

Momentum management is integrated into the allocation layer, not a separate control mode.

```
function allocate_with_desaturation(tau_desired, B_tau, u_min, u_max,
                                     state, config) -> AllocationOutput:

  if not config.enable_desaturation:
    return allocate(tau_desired, B_tau, u_min, u_max, config)

  match config.desaturation_config.strategy:

    'nullspace':
      return allocate_nullspace_desat(tau_desired, B_tau, u_min, u_max,
                                      state, config)

    'weighted':
      return allocate_weighted_desat(tau_desired, B_tau, u_min, u_max,
                                     state, config)

    'scheduled':
      return allocate_scheduled_desat(tau_desired, B_tau, u_min, u_max,
                                      state, config)
```

#### Nullspace Desaturation (Overactuated Systems)

For systems with more actuator DOFs than needed (e.g., 3RW + 3MTQ = 6 DOFs for 3 torque axes), the nullspace of $B_\tau$ can be used for momentum management with zero impact on the primary torque objective.

```
function allocate_nullspace_desat(tau_desired, B_tau, u_min, u_max,
                                  state, config) -> AllocationOutput:
  # First, solve the primary allocation
  primary_result = allocate(tau_desired, B_tau, u_min, u_max, config)
  u_primary = primary_result.u

  # Compute nullspace of B_tau
  # B_tau is [3 × n], so nullspace has dimension n-rank(B_tau)
  (U, S, Vt) = svd(B_tau)
  rank = count(S > 1e-10)
  N = Vt[rank:, :].T   # columns span nullspace, [n × (n-rank)]

  if N.shape[1] == 0:
    # No nullspace — can't desaturate without affecting torque
    return primary_result

  # Desaturation objective: drive RW momentum toward target
  # h_rw_error = h_rw_current - h_rw_target
  # We want a torque on the wheels that reduces this error:
  # tau_desat_rw = -k_desat * h_rw_error (applied to wheel subset)
  #
  # But we express this as a desired change in the full command vector u,
  # then project onto the nullspace.

  h_rw_error = state.h_rw - config.desaturation_config.h_rw_target
  # Map desired wheel desat torque to full command vector
  # (zero for non-RW actuators)
  u_desat_desired = zeros(n_total)
  u_desat_desired[rw_indices] = -k_desat * h_rw_error  # simplified

  # Project onto nullspace
  u_desat_null = N @ (N.T @ u_desat_desired)

  # Add to primary, respecting bounds
  u_combined = u_primary + u_desat_null
  u_combined = clip(u_combined, u_min, u_max)

  tau_achieved = B_tau @ u_combined
  # Note: clipping may have moved us out of the nullspace slightly
  # Verify the primary torque is still achieved
  direction_error = arccos(clamp(
    dot(tau_achieved, tau_desired) / (norm(tau_achieved) * norm(tau_desired) + 1e-12),
    -1, 1))

  return AllocationOutput(
    u = u_combined,
    tau_achieved = tau_achieved,
    alpha = primary_result.alpha,
    direction_error = direction_error,
    feasible = primary_result.feasible
  )
```

#### Weighted Desaturation

Blend desaturation into the QP cost function. Trades pointing accuracy for momentum management.

```
function allocate_weighted_desat(tau_desired, B_tau, u_min, u_max,
                                  state, config) -> AllocationOutput:
  # Formulation:
  #   min  ||B_tau @ u - tau_desired||² + w_desat * ||A_rw @ u_rw - tau_desat||²
  #   s.t. u_min <= u <= u_max
  #
  # where tau_desat = -k_desat * h_rw_error is the desired desaturation torque
  # and w_desat is the relative weight

  w_desat = config.desaturation_config.desat_weight
  h_rw_error = state.h_rw - config.desaturation_config.h_rw_target
  tau_desat = -k_desat * h_rw_error

  # Build augmented cost:
  # [ B_tau            ] u  ≈  [ tau_desired ]
  # [ sqrt(w) * A_desat]       [ sqrt(w) * tau_desat ]
  #
  # where A_desat extracts the RW portion and maps to RW torque

  A_desat = zeros(3, n_total)
  A_desat[:, rw_indices] = A_rw  # wheel axes

  A_aug = vertical_stack(B_tau, sqrt(w_desat) * A_desat)
  b_aug = concatenate(tau_desired, sqrt(w_desat) * tau_desat)

  u_opt = bounded_least_squares(A_aug, b_aug, u_min, u_max)

  tau_achieved = B_tau @ u_opt
  direction_error = arccos(clamp(
    dot(tau_achieved, tau_desired) / (norm(tau_achieved) * norm(tau_desired) + 1e-12),
    -1, 1))

  return AllocationOutput(
    u = u_opt,
    tau_achieved = tau_achieved,
    alpha = dot(tau_achieved, tau_desired / norm(tau_desired)) / norm(tau_desired),
    direction_error = direction_error,
    feasible = norm(tau_achieved - tau_desired) < 1e-6
  )
```

#### Scheduled Desaturation (Underactuated Systems)

For systems like 3MTQ+1RW where desaturation can't always happen (MTQ authority varies with B field), schedule desaturation during favorable periods.

```
function allocate_scheduled_desat(tau_desired, B_tau, u_min, u_max,
                                   state, config) -> AllocationOutput:
  # Determine if current MTQ authority is sufficient for desaturation
  # Use the degree of actuation authority metric or similar

  mtq_authority = compute_mtq_authority(state.B_body, mtq_axes)
  threshold = config.desaturation_config.desat_authority_threshold

  if mtq_authority >= threshold AND norm(state.h_rw) > h_rw_threshold:
    # Favorable period: add desaturation torque to desired
    h_rw_error = state.h_rw - config.desaturation_config.h_rw_target
    tau_desat = -k_desat * h_rw_error
    tau_combined = tau_desired + tau_desat
    # Note: this modifies the desired torque, potentially affecting pointing
    # The trade-off is controlled by k_desat
    return allocate(tau_combined, B_tau, u_min, u_max, config)
  else:
    # Unfavorable period: pointing only, no desaturation
    return allocate(tau_desired, B_tau, u_min, u_max, config)
```

---

### STEP 4: Actuator Failure Handling

When an actuator fails, the allocation adapts automatically by updating $B_\tau$ and the bounds.

```
function handle_actuator_failure(actuator_set, failed_indices) -> ActuatorSet:
  # Zero out the columns of B_tau corresponding to failed actuators
  # Set bounds to zero for failed actuators
  # The control law is unaffected — only the allocation changes

  for idx in failed_indices:
    actuator_set.u_min[idx] = 0
    actuator_set.u_max[idx] = 0
    # B_tau column is effectively ignored since u is forced to 0

  WARN "Actuator(s) {} failed — allocation adapted, control law unchanged".format(failed_indices)

  return actuator_set

  # NOTE: The control law never knows about the failure.
  # Performance degrades gracefully to whatever the remaining
  # actuators can achieve. This is a key feature of the modular pipeline.
```

---

## Top-Level Allocation Function

```
function allocation_step(tau_comp, actuator_set, state, alloc_config) -> AllocationOutput:

  # ================================================================
  # STEP 1: ASSEMBLE B_tau
  # ================================================================
  (B_tau, u_min, u_max) = assemble_B_tau(actuator_set, state)

  # ================================================================
  # STEP 2: ALLOCATE (with optional desaturation)
  # ================================================================
  if alloc_config.enable_desaturation:
    result = allocate_with_desaturation(
      tau_comp, B_tau, u_min, u_max, state, alloc_config)
  else:
    result = allocate(tau_comp, B_tau, u_min, u_max, alloc_config)

  # ================================================================
  # STEP 3: UPDATE STATE
  # ================================================================
  # For reaction wheels: update stored momentum based on commanded torque
  # h_rw += A_rw @ u_rw * dt  (integrated by the dynamics, not here)
  # But the allocation should record what was commanded for bookkeeping

  # ================================================================
  # OUTPUT
  # ================================================================
  return result
```

---

## Warnings

```
# Allocation warnings
if result.alpha < 0.5 persistently:
  WARN "Actuators achieving less than 50% of requested torque — consider reducing control gains or checking actuator sizing"

if result.direction_error > 0.1 (rad, ~6 deg) for QP:
  WARN "QP allocation direction error > 6° — consider LP or constrained QP for Lyapunov-based laws"

if B_tau rank < 3:
  INFO "Torque effectiveness matrix is rank-deficient (rank {}) — underactuated at this instant"

if config.method == 'lp' AND lp_had_to_project:
  INFO "LP projected desired torque onto achievable subspace — desired direction was not achievable"
```

---

## Notes on Solver Implementation

**LP solver**: Standard LP solvers (e.g., `scipy.optimize.linprog` with HiGHS, or GLPK) work. The problem has $n+1$ variables ($\mathbf{u}$ plus $\alpha$), 3 equality constraints, and $2n+1$ bound constraints. For typical ADCS ($n \leq 6$), this solves in microseconds.

**QP solver**: Bounded least-squares (`scipy.optimize.lsq_linear` with BVLS or TRF method) is efficient for the box-constrained QP. For the constrained QP (with direction equality), a general QP solver (OSQP, qpOASES) is needed.

**Embedded considerations**: For flight software, the LP and QP can be implemented with simple active-set methods. The problem size is small enough ($n \leq 6$) that even naive implementations run in well under 1 ms on typical satellite processors. Pre-compiled solver libraries are available for C/C++.
