# Planner Conditioning Improvements

This document outlines the conditioning improvements available in the trajectory planner and how to use them.

## Current Formulation

### Control Costs

**MTQ Cost (on magnetic dipole moment)**:
```
J_mtq = Σ mᵢ² * w_mtq
```

This penalizes the magnetic dipole moment directly, which is correct because:
- MTQs draw power proportional to the dipole moment, not the produced torque
- Even when `m ∥ B` (zero torque), the MTQ still consumes power
- Penalizing only torque would allow "free" dipole along B, wasting power

**RW Cost (on torque command)**:
```
J_rw = Σ τᵢ² * w_rw
```

### Normalized Control Cost Formulation

The `NormalizedSettingsConverter` automatically scales control costs by actuator limits:

```
w_raw = w_normalized * global_scale / u_max²
```

This ensures that:
1. `w_normalized` represents "cost of using actuator at 100% capacity"
2. The Quu condition number depends only on the ratio of normalized costs
3. Same tuning works across different actuator hardware

**Example**: With `mtq_cost=1.0` and `rw_torque_cost=5.0`, using the RW at full capacity is 5x more expensive than using the MTQ at full capacity, regardless of their physical limits.

---

## State Cost Normalization (Optional)

### The Problem

States have very different scales:
- Angular velocity `ω`: ~0.01-0.1 rad/s
- Quaternion error: ~0.1-1.0 (geodesic angle in radians)
- RW momentum `h`: ~0.001-0.01 Nms

With raw weights, the Qxx Hessian can have eigenvalues spanning orders of magnitude.

### The Solution: Scale Normalization

Enable with `use_scale_normalization=True` in `NormalizedStateCosts`:

```python
state_costs = NormalizedStateCosts(
    angle_cost=100.0,            # Cost when error = angle_scale
    angle_terminal_cost=1000.0,
    ang_vel_cost=100.0,          # Cost when rate = ang_vel_scale  
    ang_vel_terminal_cost=1000.0,
    use_scale_normalization=True,  # Enable normalization
    angle_scale_deg=90.0,          # Reference angle
    ang_vel_scale_deg_s=10.0,      # Reference angular velocity
)
```

The raw weights are computed as:
```
w_raw = w_normalized / scale²
```

So the cost becomes:
```
J = w_normalized * (state / scale)²
```

### When to Use

- **Default (legacy mode)**: Use for compatibility with existing tuning
- **Normalized mode**: Use when experiencing convergence issues or when states have very different magnitudes

### Available Presets

```python
from ADCS.controller.helpers import create_planner_settings

# Standard presets (legacy mode)
settings = create_planner_settings(satellite, preset='mtq_only')
settings = create_planner_settings(satellite, preset='mtq_plus_rw')
settings = create_planner_settings(satellite, preset='rw_primary')

# Normalized presets (experimental, better conditioning)
settings = create_planner_settings(satellite, preset='mtq_only_normalized')
settings = create_planner_settings(satellite, preset='mtq_plus_rw_normalized')
```

---

## Diagnostic Information

Use the `verbose=True` flag to see conditioning estimates:

```python
settings = create_planner_settings(satellite, preset='mtq_plus_rw', verbose=True)
```

Output:
```
============================================================
Normalized Planner Settings - Diagnostic Info
============================================================
Number of actuators: 4
Estimated Quu condition number: 5.0
Estimated Qxx condition number: 10.0

Scaling factors:
  Control scales: [0.15 0.15 0.15 0.00375]
  State normalization: disabled (legacy mode)

Computed raw weights:
  mtq_control_weight: 4.4444e+01
  rw_control_weight: 3.5556e+05
  ...
============================================================
```

### Interpreting Condition Numbers

- **Quu condition < 100**: Well-conditioned controls
- **Quu condition 100-1000**: May need more iterations
- **Quu condition > 1000**: Consider adjusting cost ratios

- **Qxx condition < 100**: Well-conditioned states
- **Qxx condition > 100**: Consider enabling scale normalization

---

## Code Locations

- `ADCS/controller/helpers/normalized_settings.py`: Normalized cost definitions
- `ADCS/controller/helpers/planner_factory.py`: Factory functions and presets
- `ADCS/controller/helpers/planner_settings.py`: Raw planner settings
- `trajectory_planner/src/planner/Satellite.cpp`: C++ cost computation

---

## Future Improvements

Potential improvements not yet implemented:

1. **Soft constraints**: Replace hard inequality constraints with smooth penalties
2. **Adaptive regularization**: Adjust regularization based on local Hessian conditioning
3. **Preconditioned line search**: Scale search direction by inverse Hessian approximation
