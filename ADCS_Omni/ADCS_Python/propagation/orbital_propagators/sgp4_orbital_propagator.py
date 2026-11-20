class SGP4OrbitalPropagator:
    def __init__(self, dt, tle=None):
        self.dt = dt
        self.tle = tle

    def step(self, state, environment, dynamics):
        raise NotImplementedError
