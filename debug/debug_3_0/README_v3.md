# User-facing API v3 prototype

This is a proposed public composition API layered over the block experiment.

```text
Satellite   = spacecraft hardware, onboard software, and initial state
Environment = shared external world and infrastructure
Scenario    = duration, numerics, kernel, and run type
Simulation  = composes the above into a runtime graph
```

`Scenario.kernel` is exactly one of `"Python"`, `"Numba"`, or `"JAX"`.
`Scenario.run` is exactly one of `"Single"` or `"Monte Carlo"`.

Run the demonstrations:

```bash
venv/bin/python -m debug.debug_3_0.run_v3 all
```

Outputs are written to `output_v3/`.

## Current lowering boundary

The canonical single-satellite graph is now lowered completely by both compiled
backends: orbit propagation, gravity/drag and gravity-gradient disturbances,
noisy sensors, estimator, controller, actuator noise, and attitude
propagation. The array state is structure-of-arrays across Monte-Carlo members,
so the same kernel handles single runs and batches.

The Python backend still supports the hierarchical formation graph. Numba/JAX
currently reject that multi-satellite graph rather than silently lowering only
part of it. Arbitrary user-defined Python blocks also remain Python-backend
blocks until they have an explicit array-lowering rule; this is the boundary
that will eventually become a formal lowering protocol.
