# ALTRO Performance Optimization Notes

## Overview
These are proposed optimizations for the hot paths in OldPlanner.cpp.
Apply incrementally and benchmark after each change.

## Files to modify:
- OldPlanner.cpp (backward pass, forward pass)
- Satellite.hpp/cpp (add cached computation methods)
- GeneralUtil.hpp (inline small functions)
- PlannerUtil.hpp (add DynamicsInfoStruct)

---

## Priority 1: Easy wins with minimal risk

See `OptimizedBackwardPass.cpp.proposed` for implementation.
