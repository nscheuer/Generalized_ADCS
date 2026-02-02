# Monte Carlo Results - Final Summary

**Date:** February 1, 2026 (Updated)
**Settings:** Optimized planner with tuning presets (`create_optimized_planner_settings`)

## Configuration

### Tuning Presets Available

Two tuning presets are available, selected via `tuning` parameter:

#### 1. `tuning="anti_spin"` (RECOMMENDED)
**Best for: Smooth trajectories with minimal spinning/oscillation**

| Metric | Value |
|--------|-------|
| Final error | 0.2°±0.2° |
| Mean error (last 500s) | 23.4°±20.7° |
| Max angular rate | 1.1°/s±0.1 |

Settings:
- Terminal angle cost: **25x** base
- Terminal ang_vel cost: **100x** base (KEY: very high to prevent spinning)
- Running angle cost: **5x** base
- Running ang_vel cost: **100x** base (KEY: very high to prevent spinning)
- MTQ control weight: **0.1x** base
- **bdot_on=1** (B-dot initialization)

#### 2. `tuning="smooth"` (Original)
**Best for: Fast convergence, may have oscillation issues**

| Metric | Value |
|--------|-------|
| Final error | 5.8°±10.4° |
| Mean error (last 500s) | 30.9°±28.6° |
| Max angular rate | 2.3°/s±1.1 |

Settings:
- Terminal angle cost: 50x base
- Terminal ang_vel cost: 25x base
- Running angle cost: 2.5x base
- Running ang_vel cost: 2.5x base
- MTQ control weight: 0.1x base

⚠️ **WARNING:** "smooth" tuning can produce trajectories that reverse mid-maneuver (error spikes back up), causing large angular rate spikes (>4°/s).

### Common Settings
- **Gauss-Newton (Hessians OFF)** - more stable than full Newton
- **Scale normalization enabled** - consistent conditioning across satellites

### Test Parameters
- **Duration:** 1000s
- **Slew angle:** 180° (maximum difficulty)
- **Runs:** 100 per configuration
- **Orbit:** Random circular, 7000km altitude, J2 enabled

## Results: Full Attitude (180° Quaternion Slew)

| Configuration | Valid | Mean Error | Median | <1° | <5° | <10° |
|---------------|-------|------------|--------|-----|-----|------|
| **3MTQ+0RW** | 100/100 (100%) | 3.18°±10.99° | 0.36° | 71% | 89% | 92% |
| **3MTQ+1RW** | 100/100 (100%) | 0.12°±0.26° | 0.03° | 97% | 100% | 100% |

### Observations

1. **100% trajectory generation success** - All 200 runs produced valid trajectories

2. **MTQ-only (3MTQ+0RW):**
   - Excellent median performance (0.36°)
   - High std (10.99°) due to ~8% outliers with >10° error
   - Outliers likely caused by unfavorable B-field geometry during slew

3. **MTQ+RW (3MTQ+1RW):**
   - Near-perfect accuracy (97% <1°, 100% <10°)
   - RW provides continuous torque authority regardless of B-field

## New Features Implemented

### 1. bdot_on modes (for initial trajectory)
| Mode | Description | Best For |
|------|-------------|----------|
| 0 | Random small controls | Baseline |
| 1 | B-dot damping | **Fixed quaternion goals (recommended)** |
| 2 | smartbdot → fallback to bdot | Changing goals |
| 3 | smartbdot + noise → fallback | Changing goals |
| **4** | **PD control (new)** | Alternative to bdot |
| **5** | **PD + noise (new)** | Exploration |

### 2. Multi-start optimization
- Runs multiple Pass 1 attempts with different initializations
- Picks best by cost before running Pass 2
- **Overhead:** ~38% more time for 4x exploration
- **Usage:** `create_optimized_planner_settings(..., use_multistart=True)`
- **Finding:** Not needed with optimized settings - bdot_on=1 already converges well

### 3. Properly scaled PD gains
PD control now scales gains to MTQ torque capability:
```cpp
double Kp = 0.3 * tau_max_approx / (J_trace / 3.0);
double Kd = 10.0 * Kp;
```

### 4. Timing instrumentation
Pass 1 vs Pass 2 breakdown now printed:
- Pass 1 (coarse): ~11.5% of total time
- Pass 2 (fine): ~88.5% of total time

## Files Modified

- `trajectory_planner/src/planner/OldPlanner.cpp` - PD modes, multi-start, timing
- `trajectory_planner/src/planner/PyPlanner.cpp` - Python bindings for multi-start
- `ADCS/controller/helpers/planner_settings.py` - multistart_modes option
- `ADCS/controller/plan_and_track_base.py` - multi-start dispatch
- `papers/Planner/mc_planner_settings.py` - create_optimized_planner_settings with multistart

## Recommended Production Settings

```python
from mc_planner_settings import create_optimized_planner_settings

# RECOMMENDED: Anti-spin tuning for smooth trajectories
settings = create_optimized_planner_settings(
    sat, 
    duration=1000,  # trajectory length in seconds
    dt_planning=1.0,  # TVLQR timestep
    tuning="anti_spin",  # RECOMMENDED for smooth motion
    use_multistart=False,  # not needed with good bdot init
    verbose=False
)
# bdot_on defaults to 1 (B-dot) which is best for fixed goals

# Alternative: Original smooth tuning (may have oscillation issues)
settings = create_optimized_planner_settings(
    sat, duration=1000, tuning="smooth"
)
```

## Tuning Comparison Visualization

![Smooth vs Anti-Spin Comparison](/tmp/smooth_vs_antispin.png)

The "smooth" tuning (blue) can produce erratic behavior:
- Error spikes back to ~180° mid-maneuver
- Angular rate exceeds 4.5°/s
- Velocity components oscillate

The "anti_spin" tuning (red) produces much better behavior:
- Monotonic error decrease
- Max rate stays under 1.2°/s
- Smooth, predictable motion
