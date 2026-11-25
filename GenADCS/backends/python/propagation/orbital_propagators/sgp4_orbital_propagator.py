from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.state import State


class SGP4OrbitalPropagator:
    def __init__(self, dt: float, tle: tuple[str, str] | None = None) -> None:
        self.dt = dt
        self.tle = tle

    def step(
        self, state: State, environment: Environment, dynamics: DynamicsConfig
    ) -> State:
        raise NotImplementedError
