# Z_BLOCKY

Z_BLOCKY is a prototype block simulator for Generalized ADCS. `Block` is a
periodically released event block; its inputs retain their latest values.
`QueriedBlock` provides on-demand request/reply methods for environmental
models such as gravity and disturbances.

Three schedulers are available:

- `BaseScheduler()` runs each block at its own `frequency_hz`. Releases use
  independent simulation timestamps. A block's execution time and signal
  delay determine when its outputs become available. Releases that occur while
  the block is still executing are skipped.
- `AsynchronousScheduler()` uses the same asynchronous timing. For a block
  marked `sample_on_demand=True`, each nominal tick records the inputs available
  at that instant and draws its completion delay. The block's `step()` runs
  only when a consumer needs that completed capture. This also works through
  chains of marked blocks: a capture can hold a reference to an upstream
  capture, which is evaluated only if the downstream capture is used. A late
  capture cannot replace a newer completed one. Snapshots and references are
  released when their capture is evaluated or discarded.
- `AcademicScheduler()` finds the lowest frequency among all event blocks and
  runs every event block at that common rate. Computation, delay, and delay
  noise are ignored; same-time feed-forward dependencies run in topological
  order. Queried blocks are excluded from the rate calculation and still answer
  queries when called.

`sample_on_demand=True` is safe for pure or stateless producers. Captured
ordinary inputs are copied so later mutation cannot change their values.
Stateful processes must not be decimated unless their skipped evolution is represented
by an explicit fast-forward/aggregation contract. A random walk, for example,
must advance its state over the skipped interval.

## Continuous state and delayed sensors

`State` holds a plant's initial vector. `Dynamics` supplies its derivative, and
`Plant.set_dynamics()` associates the two. The scheduler creates a
`ContinuousRuntime` per plant and advances it with SciPy `solve_ivp` up to the
next event. A control output connected through `Plant.bind_control()` changes
the derivative only when the output is published; controls are held between
publications. The dynamics engine is separate from the periodic `Block` class:
it has no nominal frequency or execution delay. `StateSampledBlock` is a
sensor block that reads the continuous state at a capture timestamp.

For an asynchronous state sampled sensor, the state at the capture time joins
the other captured inputs. At a consumer release, the scheduler selects the
newest capture whose completion time has passed and evaluates it once. If a
newer capture is still pending, the consumer sees the previous completed value.
`SchedulerResult.captures` exposes capture and completion times; evaluations
are in `SchedulerResult.executions` with `sampled_at_s` filled in. Use
`trace=False` in `run()` to omit these event traces for long or memory-sensitive
runs while retaining counts and final outputs.

Choose `history="dense"` to keep recent solver interpolation intervals, or
`history="cache"` to save exact state snapshots at nominal sensor times.
Both histories are pruned after samples are consumed. The cache option is a
good default for fixed rate sensors; dense output also supports queries at
arbitrary past times. Gaussian delays are unbounded, so there is no fixed
history horizon: unusually late pending samples can extend retention until a
newer completed sample supersedes them. The first continuous prototype supports one
integrated state vector per plant.

Deferred blocks may consume other blocks, including other deferred blocks, in
an acyclic graph. Blocks with direct effects on plant dynamics execute at
their nominal rate. A deferred block with a timed `QueriedBlock` dependency is
rejected because its query latency cannot be known at capture time.

`Simulation.timeline(result, output="path/stem")` saves a PNG and SVG timing
diagram across all plants. Purple is continuous integration, gray marks
unused delayed events, and green marks executed events. Each diagonal runs
from the event trigger to its completion time. Without `output`, it returns a
Matplotlib figure for further customization.

All schedulers use a heap-based event queue where applicable, flat integer
connection routes, and latest-value input slots. The Python block calls remain
the main per-execution cost; the scheduler avoids polling blocks between
scheduled events.

Run the demonstrations from the repository root:

```bash
venv/bin/python Z_BLOCKY/examples/1_block_definition.py
venv/bin/python Z_BLOCKY/examples/2_simulation_assembly.py
venv/bin/python Z_BLOCKY/examples/3_schedulers.py
venv/bin/python Z_BLOCKY/examples/4_continuous_state.py
venv/bin/python Z_BLOCKY/examples/5_benchmark.py
```

The diagrams use the Python `graphviz` package and the Graphviz `dot`
executable.
