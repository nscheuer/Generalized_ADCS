"""Small hierarchical block engine for the ADCS 3.0 architecture experiment.

The public model is deliberately Pythonic.  Composite blocks provide Simulink-
like hierarchy, while execution is always performed on a validated flat plan.
The fixed backend uses integer ticks, zero-delay publication, explicit held
feedback edges, and conservative dead-sample elimination.
"""

from __future__ import annotations

import heapq
import math
import time
import copy
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Clock:
    """Requested execution clock.  Frequencies must lie on a common lattice."""

    hz: float
    phase: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.hz) or self.hz <= 0:
            raise ValueError("clock frequency must be finite and positive")
        if not math.isfinite(self.phase) or self.phase < 0:
            raise ValueError("clock phase must be finite and non-negative")

    @property
    def period(self) -> float:
        return 1.0 / self.hz


@dataclass(frozen=True)
class Port:
    name: str
    dtype: type = object


class Block:
    """Leaf block API.

    ``sample_on_demand`` means executions whose result cannot be observed may
    be removed.  Such a block must advance skipped stochastic state from the
    supplied ``dt``; it must not contain hidden externally visible side effects.
    """

    def __init__(
        self,
        name: str,
        hz: float,
        *,
        inputs: Iterable[Port] = (),
        outputs: Iterable[Port] = (),
        phase: float = 0.0,
        sample_on_demand: bool = False,
    ) -> None:
        if "." in name or not name:
            raise ValueError("block names must be non-empty and cannot contain '.'")
        self.name = name
        self.clock = Clock(hz, phase)
        self.inputs = {port.name: port for port in inputs}
        self.outputs = {port.name: port for port in outputs}
        self.sample_on_demand = sample_on_demand
        self.parent: CompositeBlock | None = None
        self._last_execution: float | None = None

    @property
    def path(self) -> str:
        names = [self.name]
        parent = self.parent
        while parent is not None and parent.parent is not None:
            names.append(parent.name)
            parent = parent.parent
        return ".".join(reversed(names))

    def initial_outputs(self) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} must define initial_outputs()")

    def step(
        self,
        time_s: float,
        dt: float,
        inputs: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        raise NotImplementedError


class CompositeBlock:
    """Structural block containing leaves and/or other composites."""

    def __init__(self, name: str) -> None:
        if "." in name or not name:
            raise ValueError("composite names must be non-empty and cannot contain '.'")
        self.name = name
        self.parent: CompositeBlock | None = None
        self.children: list[Block | CompositeBlock] = []

    @property
    def path(self) -> str:
        names = [self.name]
        parent = self.parent
        while parent is not None and parent.parent is not None:
            names.append(parent.name)
            parent = parent.parent
        return ".".join(reversed(names))

    def add(self, child: Block | CompositeBlock) -> Block | CompositeBlock:
        if child.parent is not None:
            raise ValueError(f"{child.name!r} already belongs to a composite")
        if any(existing.name == child.name for existing in self.children):
            raise ValueError(f"duplicate child name {child.name!r} in {self.path!r}")
        child.parent = self
        self.children.append(child)
        return child


@dataclass(frozen=True)
class Endpoint:
    block: Block
    port: str

    @property
    def key(self) -> str:
        return f"{self.block.path}.{self.port}"


@dataclass(frozen=True)
class Connection:
    source: Endpoint
    target: Endpoint
    feedback: bool = False
    observer: bool = False


class Model:
    """Hierarchical model and its typed signal wiring."""

    def __init__(self, name: str = "satellite") -> None:
        self.root = CompositeBlock(name)
        self.connections: list[Connection] = []

    def connect(
        self,
        source: Block,
        output: str,
        target: Block,
        input_: str,
        *,
        feedback: bool = False,
        observer: bool = False,
    ) -> None:
        if output not in source.outputs:
            raise KeyError(f"{source.path!r} has no output {output!r}")
        if input_ not in target.inputs:
            raise KeyError(f"{target.path!r} has no input {input_!r}")
        source_type = source.outputs[output].dtype
        target_type = target.inputs[input_].dtype
        if source_type is not object and target_type is not object and source_type != target_type:
            raise TypeError(
                f"cannot connect {source.path}.{output} ({source_type.__name__}) to "
                f"{target.path}.{input_} ({target_type.__name__})"
            )
        endpoint = Endpoint(target, input_)
        if any(c.target == endpoint for c in self.connections):
            raise ValueError(f"input {endpoint.key!r} is already connected")
        self.connections.append(Connection(Endpoint(source, output), endpoint, feedback, observer))

    def leaves(self) -> list[Block]:
        leaves: list[Block] = []

        def visit(node: Block | CompositeBlock) -> None:
            if isinstance(node, Block):
                leaves.append(node)
            else:
                for child in node.children:
                    visit(child)

        visit(self.root)
        return leaves


@dataclass(frozen=True)
class BlockPlan:
    block: Block
    requested_ticks: int
    effective_ticks: int
    phase_ticks: int

    @property
    def requested_hz(self) -> float:
        return self.block.clock.hz


@dataclass
class ExecutionPlan:
    model: Model
    base_tick: float
    blocks: list[BlockPlan]
    inputs: dict[Block, dict[str, str]]
    feedback_inputs: dict[Block, set[str]]
    initial_signals: dict[str, Any]

    def for_block(self, block: Block) -> BlockPlan:
        return next(plan for plan in self.blocks if plan.block is block)


def _gcd_many(values: Iterable[int]) -> int:
    return reduce(math.gcd, values)


def _topological_order(blocks: list[Block], connections: list[Connection]) -> list[Block]:
    # Held feedback does not constrain same-timestamp execution. Observer edges
    # still place recorders after their producers, but do not create rate demand.
    adjacency = {block: set() for block in blocks}
    indegree = {block: 0 for block in blocks}
    for connection in connections:
        if connection.feedback:
            continue
        source, target = connection.source.block, connection.target.block
        if source is target or target in adjacency[source]:
            continue
        adjacency[source].add(target)
        indegree[target] += 1
    ready = sorted((b for b in blocks if indegree[b] == 0), key=lambda b: b.path)
    ordered: list[Block] = []
    while ready:
        block = ready.pop(0)
        ordered.append(block)
        for target in sorted(adjacency[block], key=lambda b: b.path):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(key=lambda b: b.path)
    if len(ordered) != len(blocks):
        cyclic = sorted(block.path for block, degree in indegree.items() if degree)
        raise ValueError(
            "zero-delay algebraic loop detected among " + ", ".join(cyclic) +
            "; mark a state-holding connection as feedback=True"
        )
    return ordered


def compile_fixed(model: Model) -> ExecutionPlan:
    """Validate and lower a hierarchy to a deterministic fixed-step plan."""

    leaves = model.leaves()
    if not leaves:
        raise ValueError("model contains no executable blocks")
    if len({block.path for block in leaves}) != len(leaves):
        raise ValueError("block paths must be unique")

    incoming: dict[Block, dict[str, str]] = {block: {} for block in leaves}
    feedback: dict[Block, set[str]] = {block: set() for block in leaves}
    consumers: dict[Block, list[Block]] = {block: [] for block in leaves}
    for connection in model.connections:
        incoming[connection.target.block][connection.target.port] = connection.source.key
        if connection.feedback:
            feedback[connection.target.block].add(connection.target.port)
        if not connection.observer:
            consumers[connection.source.block].append(connection.target.block)
    for block in leaves:
        missing = set(block.inputs) - set(incoming[block])
        if missing:
            raise ValueError(f"unconnected inputs on {block.path}: {sorted(missing)}")

    periods_ns = [int(round(block.clock.period * 1e9)) for block in leaves]
    if any(period <= 0 for period in periods_ns):
        raise ValueError("clock periods are below one nanosecond")
    # Fixed mode deliberately aligns all clocks to t=0. Clock phases are an
    # asynchronous-backend concern and do not create artificial fixed ticks.
    base_ns = _gcd_many(periods_ns)
    base_tick = base_ns / 1e9
    requested = {block: periods_ns[i] // base_ns for i, block in enumerate(leaves)}

    # Demand propagates backwards through sample-on-demand chains.  Slower
    # producers retain their native clock; faster producers inherit the fastest
    # non-observer consumer clock.  The fixed point handles nested sensor muxes.
    effective = dict(requested)
    for _ in range(len(leaves) + 1):
        changed = False
        for block in reversed(leaves):
            if not block.sample_on_demand or not consumers[block]:
                continue
            demand = min(effective[consumer] for consumer in consumers[block])
            candidate = max(requested[block], demand)
            if candidate != effective[block]:
                effective[block] = candidate
                changed = True
        if not changed:
            break

    effective_ns = {block: effective[block] * base_ns for block in leaves}
    execution_base_ns = _gcd_many(effective_ns.values())
    ordered = _topological_order(leaves, model.connections)
    plans = [
        BlockPlan(
            block,
            requested[block],
            effective_ns[block] // execution_base_ns,
            0,
        )
        for block in ordered
    ]
    initial: dict[str, Any] = {}
    for block in leaves:
        values = block.initial_outputs()
        if set(values) != set(block.outputs):
            raise ValueError(
                f"{block.path}.initial_outputs() returned {sorted(values)}, expected {sorted(block.outputs)}"
            )
        for port, value in values.items():
            dtype = block.outputs[port].dtype
            if dtype is not object and not isinstance(value, dtype):
                raise TypeError(f"initial value for {block.path}.{port} must be {dtype.__name__}")
            initial[f"{block.path}.{port}"] = value
    return ExecutionPlan(model, execution_base_ns / 1e9, plans, incoming, feedback, initial)


@dataclass(frozen=True)
class Trace:
    block: str
    time: float
    requested_hz: float
    effective_hz: float


@dataclass
class RunResult:
    backend: str
    history: dict[str, list[Any]]
    trace: list[Trace]
    wall_time: float
    base_tick: float
    execution_counts: dict[str, int]

    def array(self, name: str) -> np.ndarray:
        return np.asarray(self.history[name])


class Recorder(Block):
    """Observer block; its edges do not create producer demand."""

    def __init__(self, name: str, hz: float, ports: Iterable[Port]) -> None:
        super().__init__(name, hz, inputs=ports, outputs=())
        self.records: dict[str, list[Any]] = {port.name: [] for port in ports}
        self.records["time"] = []

    def initial_outputs(self) -> dict[str, Any]:
        return {}

    def step(self, time_s: float, dt: float, inputs: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        self.records["time"].append(time_s)
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                value = value.copy()
            self.records[name].append(value)
        return {}


class FixedStepEngine:
    """Deterministic zero-delay engine over a compiled integer clock lattice."""

    def run(self, model: Model, duration: float, *, seed: int = 0) -> RunResult:
        model = copy.deepcopy(model)
        plan = compile_fixed(model)
        rng = np.random.default_rng(seed)
        signals = dict(plan.initial_signals)
        previous_signals = dict(signals)
        trace: list[Trace] = []
        counts = {item.block.path: 0 for item in plan.blocks}
        ticks = int(math.floor(duration / plan.base_tick + 1e-9)) + 1
        started = time.perf_counter()
        for tick in range(ticks):
            time_s = tick * plan.base_tick
            # Snapshot implements explicit zero-order-held feedback at this instant.
            previous_signals.update(signals)
            for item in plan.blocks:
                relative = tick - item.phase_ticks
                if relative < 0 or relative % item.effective_ticks:
                    continue
                block = item.block
                values = {}
                for input_name, source_key in plan.inputs[block].items():
                    bank = previous_signals if input_name in plan.feedback_inputs[block] else signals
                    values[input_name] = bank[source_key]
                dt = block.clock.period if block._last_execution is None else time_s - block._last_execution
                outputs = block.step(time_s, dt, values, rng)
                if set(outputs) != set(block.outputs):
                    raise ValueError(f"{block.path}.step() returned incorrect output ports")
                for output, value in outputs.items():
                    signals[f"{block.path}.{output}"] = value
                block._last_execution = time_s
                counts[block.path] += 1
                trace.append(Trace(block.path, time_s, item.requested_hz, 1.0 / (item.effective_ticks * plan.base_tick)))
        wall = time.perf_counter() - started
        recorders = [item.block for item in plan.blocks if isinstance(item.block, Recorder)]
        history = recorders[0].records if len(recorders) == 1 else {
            f"{recorder.path}.{name}": values
            for recorder in recorders for name, values in recorder.records.items()
        }
        return RunResult("fixed-step", history, trace, wall, plan.base_tick, counts)


class AsyncEventEngine:
    """Independent-clock, zero-delay backend using the same hierarchy and blocks."""

    def run(self, model: Model, duration: float, *, seed: int = 0) -> RunResult:
        model = copy.deepcopy(model)
        plan = compile_fixed(model)  # validation and deterministic same-time order
        rng = np.random.default_rng(seed)
        signals = dict(plan.initial_signals)
        order = {item.block: i for i, item in enumerate(plan.blocks)}
        queue: list[tuple[float, int, Block]] = []
        for item in plan.blocks:
            heapq.heappush(queue, (item.block.clock.phase, order[item.block], item.block))
        trace: list[Trace] = []
        counts = {item.block.path: 0 for item in plan.blocks}
        started = time.perf_counter()
        while queue:
            time_s = queue[0][0]
            if time_s > duration + 1e-12:
                break
            simultaneous: list[Block] = []
            while queue and abs(queue[0][0] - time_s) < 1e-12:
                _, _, block = heapq.heappop(queue)
                simultaneous.append(block)
            previous_signals = dict(signals)
            for block in sorted(simultaneous, key=order.get):
                values = {}
                for input_name, source_key in plan.inputs[block].items():
                    bank = previous_signals if input_name in plan.feedback_inputs[block] else signals
                    values[input_name] = bank[source_key]
                dt = block.clock.period if block._last_execution is None else time_s - block._last_execution
                outputs = block.step(time_s, dt, values, rng)
                if set(outputs) != set(block.outputs):
                    raise ValueError(f"{block.path}.step() returned incorrect output ports")
                for output, value in outputs.items():
                    signals[f"{block.path}.{output}"] = value
                block._last_execution = time_s
                counts[block.path] += 1
                trace.append(Trace(block.path, time_s, block.clock.hz, block.clock.hz))
                heapq.heappush(queue, (time_s + block.clock.period, order[block], block))
        wall = time.perf_counter() - started
        recorders = [item.block for item in plan.blocks if isinstance(item.block, Recorder)]
        history = recorders[0].records if len(recorders) == 1 else {}
        return RunResult("asynchronous", history, trace, wall, plan.base_tick, counts)
