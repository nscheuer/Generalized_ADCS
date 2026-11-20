class KeplerianOrbitalPropagator:
    def __init__(self, dt):
        self.dt = dt

    def step(self, state, environment, dynamics):
        raise NotImplementedError
