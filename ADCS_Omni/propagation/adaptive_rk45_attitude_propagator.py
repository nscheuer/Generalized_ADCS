class AdaptiveRK45AttitudePropagator:
    def __init__(self, dt, rtol=1e-6, atol=1e-9):
        self.dt = dt
        self.rtol = rtol
        self.atol = atol

    def step(self, state, satellite, environment, dynamics):
        raise NotImplementedError
