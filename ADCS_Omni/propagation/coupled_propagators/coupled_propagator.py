class CoupledPropagator:
    def __init__(self, dt):
        self.dt = dt

    def step(self, state, satellite, environment, dynamics):
        raise NotImplementedError
