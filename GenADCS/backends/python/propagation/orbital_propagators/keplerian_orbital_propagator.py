from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.state import State


class KeplerianOrbitalPropagator:
    def __init__(self, dt: float) -> None:
        self.dt = dt

    def step(
        self, state: State, environment: Environment, dynamics: DynamicsConfig
    ) -> State:
        raise NotImplementedError
