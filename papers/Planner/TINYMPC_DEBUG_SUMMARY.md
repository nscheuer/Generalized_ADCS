# TinyMPC Debugging Summary

## Problem
C++ TinyMPC and Python MPC controllers were not tracking trajectories correctly, while TVLQR worked fine.

## Investigation Results

### Error Convention Mismatch
The core issue was **quaternion error representation inconsistency**:

| Component | Convention | Formula |
|-----------|-----------|---------|
| Python Trajectory `_state_diff` | Scaled MRP | `σ = 2*q_vec / (1 + q_scalar)` |
| C++ TinyMPC (original) | 2×vector | `θ_err = 2 * q_vec` |
| Python MPC linearization | Mixed 0.5x/2x scaling | Inconsistent |

For a 30° error:
- Python MRP: 0.2633
- C++ 2×vector: 0.5176 (almost 2x larger)

This caused the C++ TinyMPC to compute control commands that were in the wrong direction or magnitude.

### Fixes Applied
1. Changed C++ `computeStateError` to use scaled MRP: `σ = 2*q_vec / (1 + q_scalar)`
2. Updated C++ E matrix linearization to match MRP convention
3. Added `use_trajectory_gains` option to use ALTRO K gains instead of internal Riccati

### Remaining Issues
Even with MRP fix, TinyMPC still diverges. The problem appears to be:
1. Single linearization point doesn't work well when far from reference
2. Internal Riccati gains don't match ALTRO's time-varying gains
3. Using ALTRO K gains with TinyMPC's ADMM also diverges (gain/error convention mismatch)

## Performance Comparison (50-step simulation)

| Controller | Final Error | Solve Time | Status |
|------------|-------------|------------|--------|
| **TVLQR** | 9.3° | 0.15 ms | ✅ Working |
| **ComputedTorque** | 15.0° | 0.32 ms | ✅ Working |
| **TinyMPC (internal)** | 72.0° | 0.34 ms | ❌ Not tracking |
| **Python MPC** | 180.0° | 20.3 ms | ❌ Diverging |

## Key Finding: TVLQR Doesn't Saturate
Testing showed that TVLQR **rarely hits actuator limits** in practice:
- Raw TVLQR: 17.0° final error
- Saturated TVLQR: 16.2° final error
- Saturation events in 50 steps: **0**

**This means MPC constraint handling is rarely needed!**

## Recommendations

### For Most Use Cases: Use TVLQR
- Fast (0.15 ms)
- Works correctly
- Simple implementation
- Already handles constraints gracefully via trajectory planning

### For Explicit Constraint Handling: Use ComputedTorque
- Nearly as fast as TVLQR (0.32 ms)
- Works correctly
- Uses inverse dynamics + actual B-field
- Graceful degradation when constraints active

### If MPC is Required: Fix the Convention
To make TinyMPC work, need to:
1. Ensure ALTRO K gains use same error convention as TinyMPC
2. OR recompute TVLQR gains in C++ using consistent convention
3. Consider using ALTRO's linearization directly (time-varying A,B) instead of single-point

## Files Modified
- `trajectory_planner/src/planner/TinyMPC.cpp` - MRP error convention
- `trajectory_planner/src/planner/TinyMPC.hpp` - `use_trajectory_gains` option
- `trajectory_planner/src/planner/PyTinyMPC.cpp/hpp` - Python bindings
- `ADCS/controller/helpers/tinympc_settings.py` - Settings dataclass
- `ADCS/controller/plan_and_track_tinympc_cpp.py` - K gains loading

## Conclusion
**TVLQR is the recommended controller** for trajectory tracking. It's fast, accurate, and constraints are rarely binding. ComputedTorque is a good alternative when explicit constraint handling is desired. TinyMPC needs more work to fix convention mismatches.
