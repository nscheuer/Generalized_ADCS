"""User-facing v3 composition API for the block-graph prototype.

``Satellite`` owns spacecraft hardware, software, and its initial state.
``Environment`` is shared external reality.  ``Scenario`` owns numerical/run
policy.  ``Simulation`` assembles a runtime graph from those objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

from .batch_backend import run_jax_monte_carlo, run_numba_monte_carlo
from .block_engine import Block, CompositeBlock, FixedStepEngine, Model, Port, Recorder
from .prototype import initial_attitude, initial_orbit
from .satellite_model import (
    Actuation,
    Actuators,
    AttitudePropagation,
    Command,
    Controller,
    Disturbance,
    Disturbances,
    Estimate,
    OrbitPropagation,
)


class Kernel(str, Enum):
    PYTHON = "Python"
    NUMBA = "Numba"
    JAX = "JAX"


class Run(str, Enum):
    SINGLE = "Single"
    MONTE_CARLO = "Monte Carlo"


@dataclass(frozen=True)
class Integration:
    orbit_hz: float = 5.0
    attitude_hz: float = 10.0
    subsystem_hz: float = 10.0
    orbit_integrator: str = "rk4"
    attitude_integrator: str = "rk4"

    def __post_init__(self) -> None:
        if min(self.orbit_hz, self.attitude_hz, self.subsystem_hz) <= 0:
            raise ValueError("integration rates must be positive")


@dataclass(frozen=True)
class InitialState:
    orbit: np.ndarray
    attitude: np.ndarray

    @classmethod
    def leo_default(cls) -> "InitialState":
        return cls(initial_orbit(), initial_attitude())


@dataclass(frozen=True)
class Satellite:
    """Spacecraft definition including the initial state for this satellite."""

    name: str
    mass_kg: float = 12.0
    inertia_kgm2: np.ndarray = field(default_factory=lambda: np.diag([0.035, 0.045, 0.052]))
    reaction_wheels: int = 3
    magnetorquers: int = 3
    sensors: bool = True
    estimator: bool = True
    initial_state: InitialState = field(default_factory=InitialState.leo_default)

    def __post_init__(self) -> None:
        if not self.name or "." in self.name:
            raise ValueError("satellite names must be non-empty and cannot contain '.'")
        if self.sensors != self.estimator:
            raise ValueError(
                "this v3 prototype supports either sensors+estimator or neither; "
                "use state knowledge when both are disabled"
            )

    @classmethod
    def standard_cube_sat(
        cls,
        name: str,
        *,
        sensors: bool = True,
        estimator: bool = True,
        initial_state: InitialState | None = None,
    ) -> "Satellite":
        return cls(
            name=name, sensors=sensors, estimator=estimator,
            initial_state=InitialState.leo_default() if initial_state is None else initial_state,
        )


@dataclass(frozen=True)
class Environment:
    gravity: str = "J2"
    magnetic_field: str = "dipole"
    atmosphere: str = "exponential"
    ground_stations: tuple[str, ...] = ()

    @classmethod
    def standard_leo(cls, **overrides) -> "Environment":
        return cls(**overrides)


@dataclass(frozen=True)
class Scenario:
    duration: float
    kernel: Kernel | str = Kernel.PYTHON
    run: Run | str = Run.SINGLE
    integration: Integration = field(default_factory=Integration)
    monte_carlo_runs: int = 1
    seed: int = 7

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel", Kernel(self.kernel))
        object.__setattr__(self, "run", Run(self.run))
        if self.duration <= 0:
            raise ValueError("scenario duration must be positive")
        if self.monte_carlo_runs < 1:
            raise ValueError("monte_carlo_runs must be at least one")


@dataclass(frozen=True)
class GPSReading:
    position_eci_m: np.ndarray


class StateKnowledge(Block):
    """Ideal truth-to-controller interface, used when no sensors/estimator exist."""

    def __init__(self, hz: float, initial: InitialState) -> None:
        super().__init__("state_knowledge", hz, inputs=[Port("orbit", np.ndarray), Port("attitude", np.ndarray)], outputs=[Port("estimate", Estimate)])
        self.initial = initial

    def initial_outputs(self):
        return {"estimate": Estimate(self.initial.attitude[3:7].copy(), self.initial.attitude[:3].copy(), np.zeros(3))}

    def step(self, time_s, dt, inputs, rng):
        attitude = inputs["attitude"]
        return {"estimate": Estimate(attitude[3:7].copy(), attitude[:3].copy(), np.zeros(3))}


class GPS(Block):
    def __init__(self, hz: float, initial: InitialState) -> None:
        super().__init__("gps", hz, inputs=[Port("orbit", np.ndarray)], outputs=[Port("reading", GPSReading)], sample_on_demand=True)
        self.initial = initial

    def initial_outputs(self):
        return {"reading": GPSReading(self.initial.orbit[:3].copy())}

    def step(self, time_s, dt, inputs, rng):
        return {"reading": GPSReading(inputs["orbit"][:3] + rng.normal(0.0, 2.0, 3))}


class FormationController(Block):
    """Minimal shared block: deputy position relative to chief GPS position."""

    def __init__(self, hz: float) -> None:
        super().__init__(
            "formation_controller", hz,
            inputs=[Port("chief_gps", GPSReading), Port("deputy_gps", GPSReading)],
            outputs=[Port("relative_position_m", np.ndarray)],
        )

    def initial_outputs(self):
        return {"relative_position_m": np.zeros(3)}

    def step(self, time_s, dt, inputs, rng):
        return {"relative_position_m": inputs["deputy_gps"].position_eci_m - inputs["chief_gps"].position_eci_m}


class EnvironmentBlock(Block):
    """Graph-visible shared environment placeholder for the v3 visualization."""

    def __init__(self, environment: Environment) -> None:
        super().__init__("environment_models", 1.0, outputs=[Port("metadata", Environment)])
        self.environment = environment

    def initial_outputs(self):
        return {"metadata": self.environment}

    def step(self, time_s, dt, inputs, rng):
        return {"metadata": self.environment}


@dataclass
class CompiledSimulation:
    model: Model
    formation: FormationController | None = None


def _add_satellite(
    model: Model,
    parent: CompositeBlock,
    satellite: Satellite,
    integration: Integration,
    *,
    include_gps: bool = False,
) -> dict[str, Block]:
    """Add one runtime satellite graph below ``parent``."""

    orbit = parent.add(OrbitPropagation(integration.orbit_hz, integrator=integration.orbit_integrator))
    orbit.state = satellite.initial_state.orbit.copy()
    attitude = parent.add(AttitudePropagation(integration.attitude_hz, integrator=integration.attitude_integrator))
    attitude.state = satellite.initial_state.attitude.copy()
    disturbances = parent.add(Disturbances(integration.subsystem_hz))
    controller = parent.add(Controller(integration.subsystem_hz))
    actuators = parent.add(Actuators(integration.attitude_hz))
    model.connect(disturbances, "disturbance", orbit, "disturbance", feedback=True)
    model.connect(actuators, "actuation", attitude, "actuation", feedback=True)
    model.connect(disturbances, "disturbance", attitude, "disturbance", feedback=True)
    model.connect(orbit, "state", disturbances, "orbit")
    model.connect(attitude, "state", disturbances, "attitude")
    model.connect(attitude, "state", controller, "attitude")
    model.connect(orbit, "state", controller, "orbit")
    model.connect(controller, "command", actuators, "command")
    model.connect(orbit, "state", actuators, "orbit")
    model.connect(attitude, "state", actuators, "attitude")

    blocks: dict[str, Block] = {
        "orbit": orbit, "attitude": attitude, "disturbances": disturbances,
        "controller": controller, "actuators": actuators,
    }
    if satellite.sensors:
        # The v3 public API treats this as the standard onboard navigation
        # subsystem. State-knowledge is the explicit no-sensor alternative.
        from .satellite_model import SensorSuite, Estimator
        sensors = parent.add(SensorSuite(integration.subsystem_hz))
        estimator = parent.add(Estimator(integration.subsystem_hz))
        model.connect(orbit, "state", sensors, "orbit")
        model.connect(attitude, "state", sensors, "attitude")
        model.connect(sensors, "packet", estimator, "sensors")
        model.connect(orbit, "state", estimator, "orbit")
        model.connect(estimator, "estimate", controller, "estimate")
        blocks.update({"sensors": sensors, "estimator": estimator})
    else:
        knowledge = parent.add(StateKnowledge(integration.subsystem_hz, satellite.initial_state))
        model.connect(orbit, "state", knowledge, "orbit")
        model.connect(attitude, "state", knowledge, "attitude")
        model.connect(knowledge, "estimate", controller, "estimate")
        blocks["state_knowledge"] = knowledge
    if include_gps:
        gps = parent.add(GPS(integration.subsystem_hz, satellite.initial_state))
        model.connect(orbit, "state", gps, "orbit")
        blocks["gps"] = gps
    return blocks


class Simulation:
    def __init__(self, *, satellites: Mapping[str, Satellite], environment: Environment, scenario: Scenario) -> None:
        self.satellites = dict(satellites)
        self.environment = environment
        self.scenario = scenario
        if not self.satellites:
            raise ValueError("simulation requires at least one satellite")
        if any(name != satellite.name for name, satellite in self.satellites.items()):
            raise ValueError("satellite mapping keys must match Satellite.name")

    def compile(self) -> CompiledSimulation:
        model = Model("simulation")
        environment = model.root.add(CompositeBlock("environment"))
        environment.add(EnvironmentBlock(self.environment))
        if len(self.satellites) == 1:
            name, satellite = next(iter(self.satellites.items()))
            group = model.root.add(CompositeBlock(name))
            blocks = _add_satellite(model, group, satellite, self.scenario.integration)
            recorder = model.root.add(Recorder("recorder", self.scenario.integration.subsystem_hz, [
                Port("orbit", np.ndarray), Port("attitude", np.ndarray), Port("estimate", Estimate),
            ]))
            model.connect(blocks["orbit"], "state", recorder, "orbit", observer=True)
            model.connect(blocks["attitude"], "state", recorder, "attitude", observer=True)
            estimate_source = blocks.get("estimator", blocks.get("state_knowledge"))
            model.connect(estimate_source, "estimate", recorder, "estimate", observer=True)
            return CompiledSimulation(model)

        satellite_blocks: dict[str, dict[str, Block]] = {}
        for name, satellite in self.satellites.items():
            group = model.root.add(CompositeBlock(name))
            satellite_blocks[name] = _add_satellite(
                model, group, satellite, self.scenario.integration, include_gps=True,
            )
        if len(satellite_blocks) != 2:
            raise NotImplementedError("the v3 formation demonstration currently accepts exactly two satellites")
        chief_name, deputy_name = self.satellites.keys()
        formation = model.root.add(FormationController(self.scenario.integration.subsystem_hz))
        model.connect(satellite_blocks[chief_name]["gps"], "reading", formation, "chief_gps")
        model.connect(satellite_blocks[deputy_name]["gps"], "reading", formation, "deputy_gps")
        recorder = model.root.add(Recorder("recorder", self.scenario.integration.subsystem_hz, [
            Port("relative_position_m", np.ndarray),
        ]))
        model.connect(formation, "relative_position_m", recorder, "relative_position_m", observer=True)
        return CompiledSimulation(model, formation)

    def run(self):
        compiled = self.compile()
        scenario = self.scenario
        if scenario.kernel is Kernel.PYTHON:
            if scenario.run is Run.SINGLE:
                return FixedStepEngine().run(compiled.model, scenario.duration, seed=scenario.seed)
            return [
                FixedStepEngine().run(compiled.model, scenario.duration, seed=scenario.seed + run)
                for run in range(scenario.monte_carlo_runs)
            ]
        if len(self.satellites) != 1:
            raise NotImplementedError("Numba/JAX lowerings for the two-satellite formation graph are not implemented yet")
        runs = 1 if scenario.run is Run.SINGLE else scenario.monte_carlo_runs
        dt = 1.0 / scenario.integration.attitude_hz
        ratios = (
            scenario.integration.attitude_hz / scenario.integration.subsystem_hz,
            scenario.integration.attitude_hz / scenario.integration.orbit_hz,
        )
        if any(abs(ratio - round(ratio)) > 1e-12 for ratio in ratios):
            raise ValueError("compiled demo requires attitude_hz / orbit_hz and attitude_hz / subsystem_hz to be integers")
        initial = next(iter(self.satellites.values())).initial_state
        environment_degree = 4 if str(self.environment.gravity).upper() in {"J4", "J2/J3/J4", "J2+J3+J4"} else 2
        kwargs = dict(
            runs=runs, duration=scenario.duration, dt=dt, seed=scenario.seed,
            controller_stride=int(round(ratios[0])), initial_attitude_state=initial.attitude,
            initial_orbit_state=initial.orbit,
            orbit_stride=int(round(ratios[1])), subsystem_stride=int(round(ratios[0])),
            orbit_integrator=scenario.integration.orbit_integrator,
            attitude_integrator=scenario.integration.attitude_integrator,
            orbit_degree=environment_degree,
            sensors=next(iter(self.satellites.values())).sensors,
        )
        if scenario.kernel is Kernel.NUMBA:
            return run_numba_monte_carlo(**kwargs)
        return run_jax_monte_carlo(**kwargs)
