"""Environment composition container used by the force and torque models."""


class Environment:
    def __init__(self, planetary_body=None, atmosphere=None, magnetic_field=None, j_gravity=None, ephemeris=None):
        if planetary_body is None:
            from .planetary_body import Earth
            planetary_body = Earth()
        self.planetary_body = planetary_body
        self.atmosphere = atmosphere
        self.magnetic_field = magnetic_field
        self.j_gravity = j_gravity
        self.ephemeris = ephemeris


__all__ = ["Environment"]
