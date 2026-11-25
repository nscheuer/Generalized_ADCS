from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite
from GenADCS.backends.python.state import State


class AdaptiveRK45AttitudePropagator:
    def __init__(self, dt: float, rtol: float = 1e-6, atol: float = 1e-9) -> None:
        self.dt = dt
        self.rtol = rtol
        self.atol = atol

    def step(
        self,
        state: State,
        satellite: Satellite,
        environment: Environment,
        dynamics: DynamicsConfig,
    ) -> State:
        raise NotImplementedError
