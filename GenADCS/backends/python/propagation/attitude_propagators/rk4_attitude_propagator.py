from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite
from GenADCS.backends.python.state import State


class RK4AttitudePropagator:
    def __init__(self, dt: float) -> None:
        self.dt = dt

    def step(
        self,
        state: State,
        satellite: Satellite,
        environment: Environment,
        dynamics: DynamicsConfig,
    ) -> State:
        raise NotImplementedError
