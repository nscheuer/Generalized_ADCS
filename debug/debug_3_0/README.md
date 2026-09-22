# ADCS 3.0 hierarchical block prototype

This directory is an isolated architecture experiment, not yet a replacement
for `ADCS.simulate()`.  The current prototype has a Simulink-like Python model:

- typed ports and validated connections;
- composites containing arbitrary blocks or other composites;
- requested and compiled execution rates shown in the graph;
- explicit held feedback edges that break physical feedback loops;
- deterministic zero-delay fixed-step execution;
- an independent-clock asynchronous backend;
- batched Numba CPU and JAX GPU Monte Carlo kernels.

The engine is in `block_engine.py`, spacecraft blocks and composition are in
`satellite_model.py`, and compiled Monte Carlo kernels are in
`batch_backend.py`.

## Run the examples

From the repository root:

```bash
venv/bin/python -m debug.debug_3_0.run_blocks fixed
venv/bin/python -m debug.debug_3_0.run_blocks subblocks
venv/bin/python -m debug.debug_3_0.run_blocks async
venv/bin/python -m debug.debug_3_0.run_blocks cpu-mc
venv/bin/python -m debug.debug_3_0.run_blocks gpu-mc
venv/bin/python -m debug.debug_3_0.run_blocks power
venv/bin/python -m debug.debug_3_0.run_blocks benchmark
```

`all` runs every example. Plots are written to `output_v2/`. The GPU command
requires an actual JAX GPU and reports the real device; use `--allow-jax-cpu`
only to validate that backend on a machine without a GPU.

## Fixed-step semantics

`compile_fixed()` converts periods to an integer clock lattice and sorts blocks
by zero-delay dependencies. At each instant:

1. state propagators read explicitly held feedback from the previous instant;
2. feed-forward blocks execute in topological order;
3. outputs become visible immediately at the same timestamp;
4. values are held until their producer next executes.

Fixed mode aligns every clock phase to the common `t=0` origin. The asynchronous
backend preserves independently declared phases. Each run deep-copies model
state, so rerunning the same model instance with the same seed is reproducible.

Blocks marked `sample_on_demand=True` permit conservative dead-sample
elimination. A 200 Hz gyro read only by a 10 Hz estimator is compiled to 10 Hz,
while its requested rate remains 200 Hz in the model and visualization.
Stateful propagators, filters, batteries, and blocks with externally visible
side effects are never contracted. Sensor random walks advance using elapsed
time, so removing unread samples does not freeze their stochastic state.

The nested and monolithic sensor examples are numerically equivalent for the
same seed. Composite hierarchy is removed from the runtime plan and therefore
has no per-step performance cost.

Adding power is ordinary composition; the engines are unchanged:

```python
power = model.root.add(CompositeBlock("power"))
solar = power.add(SolarPanels(hz=10))
battery = power.add(Battery(hz=10))

model.connect(attitude, "state", solar, "attitude")
model.connect(orbit, "state", solar, "orbit")
model.connect(solar, "generated_w", battery, "generated_w")
model.connect(battery, "power", actuators, "power")
```

## Performance boundary

The ordinary graph backend prioritizes extension and debugging. Large Monte
Carlos lower the regular fixed-step model to structure-of-arrays kernels:

- Numba uses one compiled parallel CPU kernel, not one Python graph per run;
- JAX uses one compiled batched scan on the selected device;
- compilation and steady-state execution times are reported separately.

The lowered kernels currently demonstrate the execution architecture with the
attitude/control subset. A production implementation would give each supported
block a lowering rule and reject unsupported blocks rather than silently
changing their behavior.
