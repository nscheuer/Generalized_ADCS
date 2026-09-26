"""Periodic asynchronous and synchronous schedulers for Z_BLOCKY."""

from __future__ import annotations

import heapq
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .blocky import Block, QueriedBlock, Simulation
from .continuous import ContinuousRuntime, StateSampledBlock


@dataclass(frozen=True, slots=True)
class Execution:
    time_s: float
    block: str
    sampled_at_s: float | None = None
    completed_at_s: float | None = None


@dataclass(frozen=True, slots=True)
class SampleCapture:
    block: str
    sampled_at_s: float
    completed_at_s: float


@dataclass(slots=True, eq=False)
class _CapturedInput:
    capture: _LazyCapture
    output: str


@dataclass(slots=True, eq=False)
class _LazyCapture:
    block_id: int
    sampled_at_s: float
    completed_at_s: float
    inputs: list[Any] | None
    state_snapshot: np.ndarray | None
    rng_seed: int
    valid: bool
    outputs: Mapping[str, Any] | None = None
    refs: int = 1


@dataclass(slots=True)
class SchedulerResult:
    scheduler: str
    final_time_s: float
    executions: tuple[Execution, ...]
    execution_counts: dict[str, int]
    skipped_ticks: dict[str, int]
    latest_outputs: dict[str, Any]
    wall_time_s: float
    final_states: dict[str, np.ndarray] | None = None
    max_history_items: dict[str, int] | None = None
    captures: tuple[SampleCapture, ...] = ()


class BlockContext:
    """Per-call simulation time, environment access, and random stream."""

    __slots__ = ("time_s", "environment", "rng", "include_execution_time", "query_latency_s", "runtime", "state_override")

    def __init__(self, time_s: float, simulation: Simulation, rng: np.random.Generator, *, include_execution_time: bool, runtime: ContinuousRuntime | None = None, state_override: np.ndarray | None = None):
        self.time_s = time_s
        self.environment = simulation.environment
        self.rng = rng
        self.include_execution_time = include_execution_time
        self.query_latency_s = 0.0
        self.runtime = runtime
        self.state_override = state_override

    def state_at(self, name: str) -> np.ndarray:
        if self.runtime is None or self.runtime.state_name != name:
            raise KeyError(f"no continuous state {name!r} in this plant")
        if self.state_override is not None:
            return self.state_override.copy()
        return self.runtime.state_at(self.time_s)

    def query(self, model: str, port: str, request: Any) -> Any:
        queried: QueriedBlock = self.environment[model]
        if port not in queried.queries:
            raise KeyError(f"{model!r} has no query port {port!r}")
        response = queried.query(port, request, self)
        # Queries are synchronous from the block's point of view. In the
        # asynchronous mode, their processing time contributes to completion.
        if self.include_execution_time:
            self.query_latency_s += max(
                0.0,
                queried.execution_s + queried.delay_s + _delay_jitter(queried, self.rng),
            )
        return response


class _TimedScheduler:
    name = "base"
    include_execution_time = True
    demand_optimised = False

    def run(self, simulation: Simulation, duration_s: float, *, seed: int = 0, history: str = "cache", trace: bool = True) -> SchedulerResult:
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration_s must be finite and non-negative")
        if not simulation.blocks:
            raise ValueError("simulation contains no event-triggered blocks")

        blocks = simulation.blocks
        if history not in ("cache", "dense"):
            raise ValueError("history must be 'cache' or 'dense'")
        runtimes: dict[str, ContinuousRuntime] = {}
        for plant in simulation.plants:
            if plant.dynamics is not None:
                state = next(state for state in plant.states if state.name == plant.dynamics.state_name)
                runtimes[plant.name] = ContinuousRuntime(state, plant.dynamics, history=history)
        block_ids = simulation.block_index
        rng = np.random.default_rng(seed)
        names = [tuple(block.inputs) for block in blocks]
        port_ids = [{name: i for i, name in enumerate(block_names)} for block_names in names]
        values: list[list[Any]] = [[_MISSING] * len(block_names) for block_names in names]
        routes: list[dict[str, list[tuple[int, int]]]] = [{} for _ in blocks]
        consumers: list[list[int]] = [[] for _ in blocks]
        for source_id, output, target_id, input_name, _ in simulation.connection_index:
            input_id = port_ids[target_id][input_name]
            routes[source_id].setdefault(output, []).append((target_id, input_id))
            consumers[source_id].append(target_id)

        if self.name == "academic":
            return self._run_academic(simulation, duration_s, seed, names, port_ids, values, routes, runtimes, trace)

        return self._run_events(
            simulation, duration_s, rng, names, port_ids, values, routes, consumers, runtimes, trace,
        )

    def _run_events(self, simulation, duration_s, rng, names, port_ids, values, routes, consumers, runtimes, trace):
        blocks = simulation.blocks
        order = _topological_order(blocks, simulation.connection_index)
        rank = [0] * len(blocks)
        for i, block_id in enumerate(order):
            rank[block_id] = i

        # Heap tuple: time, phase, topological rank, insertion sequence,
        # kind, block index, payload. Arrivals at equal timestamps precede jobs.
        queue: list[tuple[float, int, int, int, int, int, Any]] = []
        sequence = 0

        def push(at, phase, rank_key, kind, block_id, payload=None):
            nonlocal sequence
            heapq.heappush(queue, (at, phase, rank_key, sequence, kind, block_id, payload))
            sequence += 1

        periods = [1.0 / block.frequency_hz for block in blocks]
        busy_until = [0.0] * len(blocks)
        executions: list[Execution] = []
        counts = {self._path(simulation, block): 0 for block in blocks}
        skipped = {self._path(simulation, block): 0 for block in blocks}
        outputs: dict[str, Any] = {}
        incoming: list[list[tuple[int, str, int]]] = [[] for _ in blocks]
        for source_id, output_name, target_id, input_name, _ in simulation.connection_index:
            incoming[target_id].append((source_id, output_name, port_ids[target_id][input_name]))
        captures: dict[int, list[_LazyCapture]] = {}
        capture_trace: list[SampleCapture] = []
        dense_pending: dict[str, set[_LazyCapture]] = {name: set() for name in runtimes}
        cached_live: dict[str, int] = {name: 0 for name in runtimes}
        latest_output_sample: dict[int, float] = {}
        control_keys = {
            (simulation.block_index[source], output): (plant.name, control)
            for plant in simulation.plants for (source, output), control in plant.controls.items()
        }
        deferred = {
            block_id for block_id, block in enumerate(blocks)
            if self.demand_optimised and block.sample_on_demand and consumers[block_id]
            and not any(key[0] == block_id for key in control_keys)
        }
        for requester_id, queried_id, _ in simulation.query_index:
            queried = simulation.queried_blocks[queried_id]
            if requester_id in deferred and (
                queried.execution_s or queried.delay_s or queried.delay_noise is not None
            ):
                raise ValueError(
                    f"{blocks[requester_id].name}: deferred queries with modeled latency are not supported"
                )

        def newest_ready(source_id: int, at: float) -> _LazyCapture | None:
            return next(
                (record for record in reversed(captures[source_id])
                 if record.valid and record.completed_at_s <= at + _EPS),
                None,
            )

        def release_inputs(record: _LazyCapture) -> None:
            if record.inputs is not None:
                saved = record.inputs
                record.inputs = None
                for item in saved:
                    if isinstance(item, _CapturedInput):
                        release_reference(item.capture)
            if record.state_snapshot is not None:
                record.state_snapshot = None
                plant_name = blocks[record.block_id]._owner.removeprefix("plant.")
                cached_live[plant_name] -= 1
            plant_name = blocks[record.block_id]._owner.removeprefix("plant.")
            dense_pending.get(plant_name, set()).discard(record)

        def release_reference(record: _LazyCapture) -> None:
            record.refs -= 1
            if record.refs == 0:
                release_inputs(record)

        def materialize(record: _LazyCapture, at: float) -> Mapping[str, Any]:
            if record.outputs is not None:
                return record.outputs
            block = blocks[record.block_id]
            if record.inputs is None or not record.valid:
                raise RuntimeError(f"discarded or invalid capture for {block.name}")
            resolved = {}
            for name, item in zip(names[record.block_id], record.inputs):
                if isinstance(item, _CapturedInput):
                    resolved[name] = materialize(item.capture, at)[item.output]
                else:
                    resolved[name] = item
            plant_name = block._owner.removeprefix("plant.")
            runtime = runtimes.get(plant_name)
            state = record.state_snapshot
            context = BlockContext(
                record.sampled_at_s, simulation, np.random.default_rng(record.rng_seed),
                include_execution_time=True, runtime=runtime, state_override=state,
            )
            result = block.step(resolved, context)
            if not isinstance(result, Mapping) or set(result) != set(block.outputs):
                actual = sorted(result) if isinstance(result, Mapping) else type(result).__name__
                raise ValueError(f"{block.name}.step() returned {actual}; expected {sorted(block.outputs)}")
            if context.query_latency_s > _EPS:
                raise ValueError(f"{block.name}: deferred query introduced unexpected latency")
            for output_name, value in result.items():
                _check_type(block.outputs[output_name].dtype, value, f"{block.name}.{output_name}")
            record.outputs = dict(result)
            path = self._path(simulation, block)
            counts[path] += 1
            if trace:
                executions.append(Execution(at, path, record.sampled_at_s, record.completed_at_s))
            if record.sampled_at_s > latest_output_sample.get(record.block_id, -1.0) + _EPS:
                latest_output_sample[record.block_id] = record.sampled_at_s
                for output_name, value in record.outputs.items():
                    outputs[f"{path}.{output_name}"] = value
            release_inputs(record)
            prune_before(record.block_id, record)
            return record.outputs

        def prune_before(source_id: int, selected: _LazyCapture) -> None:
            history = captures[source_id]
            index = history.index(selected)
            if index:
                for old in history[:index]:
                    release_reference(old)
                del history[:index]

        def prune_dense(at: float) -> None:
            for plant_name, pending in dense_pending.items():
                runtime = runtimes[plant_name]
                if runtime.history == "dense":
                    runtime.prune_before(min((item.sampled_at_s for item in pending), default=at))

        started = time.perf_counter()

        for block_id, block in enumerate(blocks):
            if block_id in deferred:
                if block.execution_s > periods[block_id] + _EPS:
                    raise ValueError(f"{block.name}: lazy sampling requires execution_s <= nominal period")
                captures[block_id] = []
                push(0.0, 1, rank[block_id], _SAMPLE_TICK, block_id)
            else:
                push(0.0, 1, rank[block_id], _PERIODIC_RELEASE, block_id)

        while queue:
            at, _, _, _, kind, block_id, payload = heapq.heappop(queue)
            if at > duration_s + _EPS:
                break
            block = blocks[block_id]
            for runtime in runtimes.values():
                runtime.advance_to(at)

            if kind == _SAMPLE_TICK:
                # Freeze the input values or references valid at this capture.
                # A reference to another deferred capture is resolved only if
                # some downstream execution eventually needs this capture.
                saved: list[Any] = [_MISSING] * len(names[block_id])
                for source_id, output_name, input_id in incoming[block_id]:
                    if source_id in deferred:
                        upstream = newest_ready(source_id, at)
                        if upstream is not None:
                            upstream.refs += 1
                            saved[input_id] = _CapturedInput(upstream, output_name)
                    elif values[block_id][input_id] is not _MISSING:
                        saved[input_id] = deepcopy(values[block_id][input_id])
                valid = all(item is not _MISSING for item in saved)
                plant_name = block._owner.removeprefix("plant.")
                runtime = runtimes.get(plant_name)
                state_snapshot = None
                if valid and isinstance(block, StateSampledBlock) and runtime.history == "cache":
                    state_snapshot = runtime.value.copy()
                    cached_live[plant_name] += 1
                    runtime.max_history_items = max(runtime.max_history_items, cached_live[plant_name])
                latency = max(0.0, block.execution_s + block.delay_s + _delay_jitter(block, rng))
                record = _LazyCapture(
                    block_id, at, at + latency, saved, state_snapshot,
                    int(rng.integers(0, 2**63)), valid,
                )
                if valid:
                    captures[block_id].append(record)
                    if isinstance(block, StateSampledBlock) and runtime.history == "dense":
                        dense_pending[plant_name].add(record)
                else:
                    release_reference(record)
                if trace:
                    capture_trace.append(SampleCapture(self._path(simulation, block), at, at + latency))
                push(at + periods[block_id], 1, rank[block_id], _SAMPLE_TICK, block_id)
                continue

            if kind == _SIGNAL_ARRIVAL:
                _, value, target_id, input_id = payload
                values[target_id][input_id] = value
                continue
            if kind == _OUTPUT_PUBLISH:
                output_key, value = payload
                outputs[output_key] = value
                control = control_keys.get((block_id, output_key.rsplit(".", 1)[-1]))
                if control is not None:
                    runtimes[control[0]].controls[control[1]] = value
                continue

            if kind == _PERIODIC_RELEASE:
                push(at + periods[block_id], 1, rank[block_id], _PERIODIC_RELEASE, block_id)

            if at + _EPS < busy_until[block_id]:
                skipped[self._path(simulation, block)] += 1
                continue

            if self.demand_optimised:
                for source_id, output_name, input_id in incoming[block_id]:
                    if source_id in deferred:
                        selected = newest_ready(source_id, at)
                        if selected is not None:
                            values[block_id][input_id] = materialize(selected, at)[output_name]
                            prune_before(source_id, selected)
                prune_dense(at)

            missing = [name for name, value in zip(names[block_id], values[block_id]) if value is _MISSING]
            if missing:
                # The periodic task still ticks, but cannot run until all its
                # input signals have produced an initial value.
                skipped[self._path(simulation, block)] += 1
                continue

            inputs = dict(zip(names[block_id], values[block_id]))
            runtime = runtimes.get(block._owner.removeprefix("plant."))
            context = BlockContext(at, simulation, rng, include_execution_time=self.include_execution_time, runtime=runtime)
            result = block.step(inputs, context)
            if not isinstance(result, Mapping) or set(result) != set(block.outputs):
                actual = sorted(result) if isinstance(result, Mapping) else type(result).__name__
                raise ValueError(f"{block.name}.step() returned {actual}; expected {sorted(block.outputs)}")

            path = self._path(simulation, block)
            counts[path] += 1
            exec_s = block.execution_s if self.include_execution_time else 0.0
            latency = exec_s + block.delay_s + _delay_jitter(block, rng) + context.query_latency_s
            latency = max(0.0, latency)
            completed_at = at + latency
            if trace:
                executions.append(Execution(at, path, completed_at_s=completed_at))
            busy_until[block_id] = at + max(exec_s, periods[block_id])

            for output_name, value in result.items():
                _check_type(block.outputs[output_name].dtype, value, f"{block.name}.{output_name}")
                output_key = f"{path}.{output_name}"
                available_at = at + latency
                for target_id, input_id in routes[block_id].get(output_name, ()):
                    if available_at <= at + _EPS:
                        values[target_id][input_id] = value
                    else:
                        push(available_at, 0, rank[target_id], _SIGNAL_ARRIVAL, target_id,
                             (output_name, value, target_id, input_id))
                if available_at <= at + _EPS:
                    outputs[output_key] = value
                    control = control_keys.get((block_id, output_name))
                    if control is not None:
                        runtimes[control[0]].controls[control[1]] = value
                else:
                    push(available_at, 0, rank[block_id], _OUTPUT_PUBLISH, block_id, (output_key, value))

        if self.demand_optimised:
            for block_id in deferred:
                nominal = int(math.floor(duration_s * blocks[block_id].frequency_hz + _EPS)) + 1
                path = self._path(simulation, blocks[block_id])
                skipped[path] = max(skipped[path], nominal - counts[path])

        simulation.time_s = duration_s
        for runtime in runtimes.values():
            runtime.advance_to(duration_s)
        return SchedulerResult(
            self.name, duration_s, tuple(executions), counts, skipped, outputs,
            time.perf_counter() - started,
            {name: runtime.value.copy() for name, runtime in runtimes.items()},
            {name: runtime.max_history_items for name, runtime in runtimes.items()},
            tuple(capture_trace),
        )

    def _run_academic(self, simulation, duration_s, seed, names, port_ids, values, routes, runtimes, trace):
        blocks = simulation.blocks
        order = _topological_order(blocks, simulation.connection_index)
        plant_blocks: dict[str, list[int]] = {plant.name: [] for plant in simulation.plants}
        for block_id, block in enumerate(blocks):
            plant_name = block._owner.removeprefix("plant.")
            plant_blocks[plant_name].append(block_id)
        plant_periods = {
            name: max(1.0 / blocks[block_id].frequency_hz for block_id in block_ids)
            for name, block_ids in plant_blocks.items() if block_ids
        }
        plant_block_sets = {name: set(block_ids) for name, block_ids in plant_blocks.items()}
        order_by_plant = {
            name: tuple(block_id for block_id in order if block_id in plant_block_sets[name])
            for name in plant_blocks
        }
        rng = np.random.default_rng(seed)
        counts = {self._path(simulation, block): 0 for block in blocks}
        skipped = {self._path(simulation, block): 0 for block in blocks}
        outputs: dict[str, Any] = {}
        executions: list[Execution] = []
        started = time.perf_counter()
        plant_queue = [
            (0.0, plant_index, name, plant_periods[name], 0)
            for plant_index, name in enumerate(plant_periods)
        ]
        heapq.heapify(plant_queue)
        while plant_queue:
            at, plant_index, plant_name, period, step = heapq.heappop(plant_queue)
            if at > duration_s + _EPS:
                continue
            runtime = runtimes.get(plant_name)
            if runtime is not None:
                runtime.advance_to(at)
            for block_id in order_by_plant[plant_name]:
                block = blocks[block_id]
                missing = [name for name, value in zip(names[block_id], values[block_id]) if value is _MISSING]
                if missing:
                    raise ValueError(f"academic step cannot initialize {block.name!r}; missing inputs {missing}")
                context = BlockContext(at, simulation, rng, include_execution_time=False, runtime=runtime)
                result = block.step(dict(zip(names[block_id], values[block_id])), context)
                if not isinstance(result, Mapping) or set(result) != set(block.outputs):
                    actual = sorted(result) if isinstance(result, Mapping) else type(result).__name__
                    raise ValueError(f"{block.name}.step() returned {actual}; expected {sorted(block.outputs)}")
                path = self._path(simulation, block)
                if trace:
                    executions.append(Execution(at, path, completed_at_s=at))
                counts[path] += 1
                for output_name, value in result.items():
                    _check_type(block.outputs[output_name].dtype, value, f"{block.name}.{output_name}")
                    outputs[f"{path}.{output_name}"] = value
                    for target_id, input_id in routes[block_id].get(output_name, ()):
                        values[target_id][input_id] = value
                    if runtime is not None:
                        control = next((control for (source, output), control in
                                        next(plant for plant in simulation.plants if plant.name == plant_name).controls.items()
                                        if source is block and output == output_name), None)
                        if control is not None:
                            runtime.controls[control] = value
            next_at = (step + 1) * period
            if next_at <= duration_s + _EPS:
                heapq.heappush(plant_queue, (next_at, plant_index, plant_name, period, step + 1))

        simulation.time_s = duration_s
        for runtime in runtimes.values():
            runtime.advance_to(duration_s)
        return SchedulerResult(
            self.name, duration_s, tuple(executions), counts, skipped, outputs,
            time.perf_counter() - started,
            {name: runtime.value.copy() for name, runtime in runtimes.items()},
            {name: runtime.max_history_items for name, runtime in runtimes.items()},
        )

    @staticmethod
    def _path(simulation: Simulation, block: Block) -> str:
        return f"{block._owner.removeprefix('plant.')}.{block.name}"


class BaseScheduler(_TimedScheduler):
    """Run each block at its own frequency on independent simulation times."""

    name = "base"


class AsynchronousScheduler(_TimedScheduler):
    """Capture marked pure blocks each tick; evaluate only observed captures."""

    name = "asynchronous"
    demand_optimised = True


class AcademicScheduler(_TimedScheduler):
    """Run all event blocks at the slowest rate with zero execution/delay."""

    name = "academic"
    include_execution_time = False


_MISSING = object()
_PERIODIC_RELEASE = 0
_SIGNAL_ARRIVAL = 1
_OUTPUT_PUBLISH = 3
_SAMPLE_TICK = 4
_EPS = 1e-12


def _delay_jitter(block: Block | QueriedBlock, rng: np.random.Generator) -> float:
    if block.delay_noise is None:
        return 0.0
    return block.delay_noise.sample(rng=rng)


def _topological_order(blocks: tuple[Block, ...], connections) -> tuple[int, ...]:
    count = len(blocks)
    outgoing = [[] for _ in blocks]
    indegree = [0] * count
    seen = set()
    for source_id, _, target_id, _, _ in connections:
        if (source_id, target_id) not in seen:
            seen.add((source_id, target_id))
            outgoing[source_id].append(target_id)
            indegree[target_id] += 1
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        source_id = heapq.heappop(ready)
        ordered.append(source_id)
        for target_id in outgoing[source_id]:
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                heapq.heappush(ready, target_id)
    if len(ordered) != count:
        raise ValueError("zero-delay event cycle in graph; feedback support is not implemented in Z_BLOCKY schedulers")
    return tuple(ordered)


def _check_type(dtype: type, value: Any, name: str) -> None:
    if dtype is not object and not isinstance(value, dtype):
        raise TypeError(f"{name} must be {dtype.__name__}, got {type(value).__name__}")
