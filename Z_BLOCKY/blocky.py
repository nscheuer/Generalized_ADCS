"""Fast assembly interfaces for event blocks and on-demand queried blocks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class Port:
    name: str
    dtype: type = object
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or "." in self.name:
            raise ValueError("port names must be non-empty and cannot contain '.'")
        if not isinstance(self.dtype, type):
            raise TypeError("port dtype must be a type")


class NoiseModel(ABC):
    @abstractmethod
    def sample(self, *, rng: Any = None) -> float:
        """Sample noise when an execution engine explicitly requests it."""


@dataclass(frozen=True, slots=True)
class GaussianNoise(NoiseModel):
    mean: float = 0.0
    standard_deviation: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.mean, Real) or not isinstance(self.standard_deviation, Real):
            raise TypeError("Gaussian noise parameters must be real numbers")
        if self.standard_deviation < 0:
            raise ValueError("Gaussian noise standard deviation cannot be negative")

    def sample(self, *, rng: Any = None) -> float:
        if rng is None:
            raise ValueError("GaussianNoise.sample() requires an explicit RNG")
        return float(rng.normal(self.mean, self.standard_deviation))


@dataclass(frozen=True, slots=True)
class QueryPort:
    """A named request/reply interface exposed by a ``QueriedBlock``."""

    name: str
    request_type: type = object
    response_type: type = object

    def __post_init__(self) -> None:
        if not self.name or "." in self.name:
            raise ValueError("query port names must be non-empty and cannot contain '.'")
        if not isinstance(self.request_type, type) or not isinstance(self.response_type, type):
            raise TypeError("query request_type and response_type must be types")


class Block(ABC):
    """Event-triggered block. Inputs use latest-value delivery by default."""

    __slots__ = (
        "name", "frequency_hz", "execution_s", "delay_s", "delay_noise",
        "inputs", "outputs", "sample_on_demand", "_owner",
    )

    def __init__(
        self,
        name: str,
        *,
        frequency_hz: float,
        execution_s: float = 0.0,
        delay_s: float = 0.0,
        delay_noise: NoiseModel | None = None,
        sample_on_demand: bool = False,
        inputs: Iterable[Port] = (),
        outputs: Iterable[Port] = (),
    ) -> None:
        _validate_name(name, "block")
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if execution_s < 0 or delay_s < 0:
            raise ValueError("execution_s and delay_s cannot be negative")
        if delay_noise is not None and not isinstance(delay_noise, NoiseModel):
            raise TypeError("delay_noise must be a NoiseModel or None")
        self.name = name
        self.frequency_hz = float(frequency_hz)
        self.execution_s = float(execution_s)
        self.delay_s = float(delay_s)
        self.delay_noise = delay_noise
        self.sample_on_demand = bool(sample_on_demand)
        self.inputs = _port_map(inputs, "input")
        self.outputs = _port_map(outputs, "output")
        self._owner: str | None = None

    @abstractmethod
    def step(self, inputs: Mapping[str, Any], context: Any = None) -> Mapping[str, Any]:
        """Handle a trigger using the latest values available on input ports."""

    def describe(self) -> str:
        input_names = ", ".join(self.inputs) or "-"
        output_names = ", ".join(self.outputs) or "-"
        noise = type(self.delay_noise).__name__ if self.delay_noise else "none"
        return (
            f"{self.name} [triggered, nominal={self.frequency_hz:g} Hz, "
            f"delay={self.delay_s:g} s, noise={noise}, "
            f"sample_on_demand={self.sample_on_demand}] "
            f"in=({input_names}) out=({output_names})"
        )


class QueriedBlock(ABC):
    """On-demand request/reply block, evaluated when a consumer queries it."""

    __slots__ = ("name", "execution_s", "delay_s", "delay_noise", "queries", "_owner")

    def __init__(
        self,
        name: str,
        *,
        queries: Iterable[QueryPort],
        execution_s: float = 0.0,
        delay_s: float = 0.0,
        delay_noise: NoiseModel | None = None,
    ) -> None:
        _validate_name(name, "queried block")
        if execution_s < 0 or delay_s < 0:
            raise ValueError("execution_s and delay_s cannot be negative")
        if delay_noise is not None and not isinstance(delay_noise, NoiseModel):
            raise TypeError("delay_noise must be a NoiseModel or None")
        query_map: dict[str, QueryPort] = {}
        for query in queries:
            if not isinstance(query, QueryPort):
                raise TypeError("queries must contain QueryPort instances")
            if query.name in query_map:
                raise ValueError(f"duplicate query port name {query.name!r}")
            query_map[query.name] = query
        if not query_map:
            raise ValueError("QueriedBlock requires at least one query port")
        self.name = name
        self.execution_s = float(execution_s)
        self.delay_s = float(delay_s)
        self.delay_noise = delay_noise
        self.queries = query_map
        self._owner: str | None = None

    @abstractmethod
    def query(self, port: str, request: Any, context: Any = None) -> Any:
        """Return the requested value for the supplied request/context."""

    def describe(self) -> str:
        ports = ", ".join(
            f"{name}({item.request_type.__name__} → {item.response_type.__name__})"
            for name, item in self.queries.items()
        )
        return f"{self.name} [queried on demand, execution={self.execution_s:g} s] queries=({ports})"


@dataclass(frozen=True, slots=True)
class Connection:
    source: Block | QueriedBlock
    output: str
    target: Block
    input: str
    kind: str = "latest"

    @property
    def label(self) -> str:
        return f"{self.source.name}.{self.output} -[{self.kind}]-> {self.target.name}.{self.input}"


@dataclass(frozen=True, slots=True)
class QueryConnection:
    requester: Block
    source: QueriedBlock
    query: str


class Model:
    """Assembly helper for ordinary event blocks within a plant."""

    def __init__(self, name: str = "model") -> None:
        _validate_name(name, "model")
        self.name = name
        self.blocks: list[Block] = []
        self.connections: list[Connection] = []

    def add(self, block: Block) -> Block:
        if not isinstance(block, Block):
            raise TypeError("Model.add() accepts event-triggered Block objects")
        if any(existing.name == block.name for existing in self.blocks):
            raise ValueError(f"duplicate block name {block.name!r}")
        self.blocks.append(block)
        return block

    def connect(self, source: Block, output: str, target: Block, input: str) -> Connection:
        if source not in self.blocks or target not in self.blocks:
            raise ValueError("both blocks must belong to this model")
        return self._connect(source, output, target, input)

    def auto_connect(self, *, strict: bool = True) -> tuple[Connection, ...]:
        created: list[Connection] = []
        errors: list[str] = []
        outputs: dict[str, list[tuple[Block, Port]]] = {}
        for block in self.blocks:
            for port in block.outputs.values():
                outputs.setdefault(port.name, []).append((block, port))
        for target in self.blocks:
            for input_name, input_port in target.inputs.items():
                if any(c.target is target and c.input == input_name for c in self.connections):
                    continue
                candidates = [
                    (source, output) for source, output in outputs.get(input_name, ())
                    if source is not target and _compatible(output.dtype, input_port.dtype)
                ]
                if len(candidates) == 1:
                    source, output = candidates[0]
                    created.append(self._connect(source, output.name, target, input_name))
                elif strict:
                    if candidates:
                        matches = ", ".join(f"{b.name}.{p.name}" for b, p in candidates)
                        errors.append(f"{target.name}.{input_name}: ambiguous matches ({matches})")
                    else:
                        errors.append(f"{target.name}.{input_name}: no matching output")
        if errors:
            for connection in created:
                self.connections.remove(connection)
            raise ValueError("automatic connection failed:\n- " + "\n- ".join(errors))
        return tuple(created)

    def _connect(self, source: Block | QueriedBlock, output: str, target: Block, input: str) -> Connection:
        if input not in target.inputs:
            raise KeyError(f"{target.name!r} has no input {input!r}")
        if any(c.target is target and c.input == input for c in self.connections):
            raise ValueError(f"input {target.name}.{input} is already connected")
        if isinstance(source, Block):
            if output not in source.outputs:
                raise KeyError(f"{source.name!r} has no output {output!r}")
            source_type = source.outputs[output].dtype
        else:
            if output not in source.queries:
                raise KeyError(f"{source.name!r} has no query port {output!r}")
            source_type = source.queries[output].response_type
        target_type = target.inputs[input].dtype
        if not _compatible(source_type, target_type):
            raise TypeError(f"incompatible connection {source.name}.{output} -> {target.name}.{input}")
        connection = Connection(source, output, target, input)
        self.connections.append(connection)
        return connection


@dataclass(frozen=True, slots=True)
class State:
    """A named plant state with an optional initial value for integration."""

    name: str
    dtype: type = object
    initial: Any = None

    def __post_init__(self) -> None:
        _validate_name(self.name, "state")
        if not isinstance(self.dtype, type):
            raise TypeError("state dtype must be a type")


class Environment:
    """Immutable collection of environmental models that answer queries."""

    __slots__ = ("name", "_models")

    def __init__(self, models: Mapping[str, QueriedBlock] | None = None, *, name: str = "environment", **named_models: QueriedBlock) -> None:
        _validate_name(name, "environment")
        merged = dict(models or {})
        overlap = set(merged).intersection(named_models)
        if overlap:
            raise ValueError(f"duplicate environment model names: {sorted(overlap)}")
        merged.update(named_models)
        if any(not isinstance(model, QueriedBlock) for model in merged.values()):
            raise TypeError("Environment contains only QueriedBlock instances")
        if any(not isinstance(key, str) or not key for key in merged):
            raise ValueError("environment model names must be non-empty strings")
        for key, model in merged.items():
            if model.name != key:
                raise ValueError(f"environment key {key!r} must match model name {model.name!r}")
            if model._owner is not None:
                raise ValueError(f"queried block {model.name!r} already belongs to {model._owner}")
            model._owner = "environment"
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_models", MappingProxyType(merged))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Environment is immutable after construction")

    @property
    def models(self) -> Mapping[str, QueriedBlock]:
        return self._models

    def __getitem__(self, name: str) -> QueriedBlock:
        return self._models[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._models.get(name, default)


class Plant:
    """One plant with mutable assembly, states, and event-triggered blocks."""

    __slots__ = ("name", "_model", "_states", "_queries", "_dynamics", "_controls", "_sealed")

    def __init__(self, name: str) -> None:
        _validate_name(name, "plant")
        self.name = name
        self._model = Model(name)
        self._states: list[State] = []
        self._queries: list[QueryConnection] = []
        self._dynamics = None
        self._controls: dict[tuple[Block, str], str] = {}
        self._sealed = False

    @property
    def blocks(self) -> tuple[Block, ...]:
        return tuple(self._model.blocks)

    @property
    def connections(self) -> tuple[Connection, ...]:
        return tuple(self._model.connections)

    @property
    def states(self) -> tuple[State, ...]:
        return tuple(self._states)

    @property
    def queries(self) -> tuple[QueryConnection, ...]:
        return tuple(self._queries)

    @property
    def dynamics(self):
        return self._dynamics

    @property
    def controls(self) -> Mapping[tuple[Block, str], str]:
        return MappingProxyType(self._controls)

    def set_dynamics(self, dynamics: Any) -> None:
        from .continuous import Dynamics

        self._assert_open()
        if not isinstance(dynamics, Dynamics):
            raise TypeError("dynamics must be a Dynamics instance")
        if dynamics.state_name not in {state.name for state in self._states}:
            raise ValueError(f"plant has no state {dynamics.state_name!r}")
        self._dynamics = dynamics

    def bind_control(self, source: Block, output: str, control: str) -> None:
        """Apply a block output to dynamics when that output is published."""
        self._assert_open()
        if source not in self._model.blocks or output not in source.outputs:
            raise ValueError("control source must be a plant block output")
        _validate_name(control, "control")
        self._controls[(source, output)] = control

    def add(self, block: Block) -> Block:
        self._assert_open()
        if block._owner is not None:
            raise ValueError(f"block {block.name!r} already belongs to {block._owner}")
        result = self._model.add(block)
        block._owner = f"plant.{self.name}"
        return result

    def add_state(self, state: State) -> State:
        self._assert_open()
        if any(existing.name == state.name for existing in self._states):
            raise ValueError(f"duplicate state name {state.name!r}")
        self._states.append(state)
        return state

    def connect(self, source: Block | QueriedBlock, output: str, target: Block, input: str) -> Connection:
        self._assert_open()
        if target not in self._model.blocks:
            raise ValueError("target block must belong to this plant")
        if isinstance(source, Block) and source not in self._model.blocks:
            raise ValueError("source block must belong to this plant")
        if isinstance(source, QueriedBlock):
            raise TypeError("QueriedBlock objects use Plant.query(), not event connections")
        return self._model._connect(source, output, target, input)

    def query(self, requester: Block, source: QueriedBlock, query: str) -> QueryConnection:
        """Declare a query dependency from a plant block to an environment model."""
        self._assert_open()
        if requester not in self._model.blocks:
            raise ValueError("query requester must belong to this plant")
        if source._owner != "environment":
            raise ValueError("queried source must belong to the Simulation Environment")
        if query not in source.queries:
            raise KeyError(f"{source.name!r} has no query port {query!r}")
        record = QueryConnection(requester, source, query)
        if record not in self._queries:
            self._queries.append(record)
        return record

    def auto_connect(self, *, strict: bool = True) -> tuple[Connection, ...]:
        self._assert_open()
        return self._model.auto_connect(strict=strict)

    def _seal(self) -> None:
        self._sealed = True

    def _assert_open(self) -> None:
        if self._sealed:
            raise RuntimeError(f"plant {self.name!r} is already part of a Simulation")


class Simulation:
    """Flattened assembly containing an Environment and one or more Plants."""

    __slots__ = (
        "name", "environment", "plants", "blocks", "queried_blocks", "states",
        "connections", "query_connections", "block_index", "connection_index", "query_index", "time_s",
    )

    def __init__(self, *, environment: Environment, plants: Iterable[Plant], name: str = "simulation", auto_connect: bool = True) -> None:
        if not isinstance(environment, Environment):
            raise TypeError("environment must be an Environment")
        plant_tuple = tuple(plants)
        if not plant_tuple or any(not isinstance(plant, Plant) for plant in plant_tuple):
            raise ValueError("simulation requires one or more Plant instances")
        _validate_name(name, "simulation")
        if len({plant.name for plant in plant_tuple}) != len(plant_tuple):
            raise ValueError("plant names must be unique within a simulation")
        if any(plant._sealed for plant in plant_tuple):
            raise ValueError("a plant can belong to only one Simulation")
        if auto_connect:
            for plant in plant_tuple:
                plant.auto_connect(strict=False)

        all_blocks = tuple(block for plant in plant_tuple for block in plant.blocks)
        from .continuous import StateSampledBlock

        for plant in plant_tuple:
            for block in plant.blocks:
                if isinstance(block, StateSampledBlock):
                    if plant.dynamics is None or plant.dynamics.state_name != block.state_name:
                        raise ValueError(f"{block.name!r} requires dynamics for state {block.state_name!r}")
        block_ids = {block: i for i, block in enumerate(all_blocks)}
        query_blocks = tuple(environment.models.values())
        all_connections = tuple(c for plant in plant_tuple for c in plant.connections)
        all_queries = tuple(c for plant in plant_tuple for c in plant.queries)
        connection_ids = []
        for connection in all_connections:
            source_id = block_ids[connection.source]
            connection_ids.append((source_id, connection.output, block_ids[connection.target], connection.input, "latest"))

        self.name = name
        self.environment = environment
        self.plants = plant_tuple
        self.blocks = all_blocks
        self.queried_blocks = query_blocks
        self.states = tuple((plant.name, state) for plant in plant_tuple for state in plant.states)
        self.connections = all_connections
        self.query_connections = all_queries
        self.block_index = block_ids
        self.connection_index = tuple(connection_ids)
        self.query_index = tuple(
            (block_ids[query.requester], query_blocks.index(query.source), query.query)
            for query in all_queries
        )
        self.time_s = 0.0
        for plant in plant_tuple:
            plant._seal()

    def diagram(self, *, show: bool = False, output: str | None = None, title: str | None = None):
        """Render nested Environment/Plant boundaries with Graphviz.

        Returns the rendered file path when ``output`` is provided; otherwise
        returns Graphviz source.  Set ``show=True`` to open the rendered file.
        Requires the small ``graphviz`` Python package and Graphviz executable.
        """

        from graphviz import Digraph

        graph = Digraph(name=self.name, format="svg", graph_attr={
            "rankdir": "LR", "bgcolor": "white", "pad": "0.25", "nodesep": "0.45",
            "ranksep": "0.7", "fontname": "Inter", "label": title or self.name,
            "labelloc": "t", "fontsize": "18",
        }, node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Inter", "fontsize": "10", "margin": "0.12,0.08"},
            edge_attr={"fontname": "Inter", "fontsize": "8", "color": "#64748b", "arrowsize": "0.7"})

        with graph.subgraph(name="cluster_environment") as env_graph:
            env_graph.attr(label=f"Environment · {self.environment.name}", color="#38bdf8", style="rounded,filled", fillcolor="#f0f9ff", fontcolor="#075985", penwidth="1.5")
            if not self.queried_blocks:
                env_graph.node("empty_environment", "No environmental models", shape="plaintext", fontcolor="#64748b")
            for model in self.queried_blocks:
                node_id = f"env_{model.name}"
                env_graph.node(node_id, f"{model.name}\nqueried on demand\n" + " · ".join(model.queries), fillcolor="#bae6fd", color="#0284c7")

        for plant in self.plants:
            with graph.subgraph(name=f"cluster_plant_{plant.name}") as plant_graph:
                plant_graph.attr(label=f"Plant · {plant.name}", color="#a78bfa", style="rounded,filled", fillcolor="#faf5ff", fontcolor="#6b21a8", penwidth="1.5")
                if not plant.states and not plant.blocks:
                    plant_graph.node(f"empty_{plant.name}", "Empty plant", shape="plaintext", fontcolor="#64748b")
                for state in plant.states:
                    plant_graph.node(f"state_{plant.name}_{state.name}", f"{state.name}\nstate · {state.dtype.__name__}", shape="cylinder", fillcolor="#ede9fe", color="#8b5cf6")
                if plant.dynamics is not None:
                    plant_graph.node(f"dynamics_{plant.name}", "continuous dynamics\nsolver + history", shape="box", fillcolor="#ddd6fe", color="#8b5cf6")
                    plant_graph.edge(f"dynamics_{plant.name}", f"state_{plant.name}_{plant.dynamics.state_name}", label="integrates", color="#8b5cf6")
                for block in plant.blocks:
                    detail = f"{block.frequency_hz:g} Hz · latest value\ndelay {block.delay_s:g} s"
                    graph_node = f"plant_{plant.name}_{block.name}"
                    plant_graph.node(graph_node, f"{block.name}\ntriggered block\n{detail}", fillcolor="#dcfce7", color="#22c55e")
                    if getattr(block, "state_name", None) is not None:
                        plant_graph.edge(f"state_{plant.name}_{block.state_name}", graph_node, label="sample", color="#8b5cf6")
                for (source, output_name), control in plant.controls.items():
                    plant_graph.edge(f"plant_{plant.name}_{source.name}", f"dynamics_{plant.name}", label=f"{output_name} → {control}", color="#8b5cf6")

        for connection in self.connections:
            source = f"plant_{connection.source._owner.removeprefix('plant.').split('.', 1)[0]}_{connection.source.name}"
            target_plant = connection.target._owner.removeprefix("plant.")
            target = f"plant_{target_plant}_{connection.target.name}"
            graph.edge(source, target, label=f"{connection.output} → {connection.input}")
        for query in self.query_connections:
            requester_plant = query.requester._owner.removeprefix("plant.")
            requester = f"plant_{requester_plant}_{query.requester.name}"
            graph.edge(requester, f"env_{query.source.name}", label=f"query: {query.query}", style="dashed", color="#0284c7", constraint="false")
        if output is None:
            return graph
        rendered = graph.render(filename=output, cleanup=True, view=show)
        return rendered

    def timeline(self, result: Any, *, output: str | None = None, show: bool = False, title: str | None = None):
        """Plot the execution history of all plants in this simulation."""
        from .plotting import plot_timeline

        return plot_timeline(self, result, output=output, show=show, title=title)


def _validate_name(name: str, kind: str) -> None:
    if not isinstance(name, str) or not name or "." in name:
        raise ValueError(f"{kind} names must be non-empty strings and cannot contain '.'")


def _port_map(ports: Iterable[Port], direction: str) -> dict[str, Port]:
    result: dict[str, Port] = {}
    for port in ports:
        if not isinstance(port, Port):
            raise TypeError(f"{direction}s must contain Port instances")
        if port.name in result:
            raise ValueError(f"duplicate {direction} port name {port.name!r}")
        result[port.name] = port
    return result


def _compatible(source: type, target: type) -> bool:
    return source is object or target is object or source == target
