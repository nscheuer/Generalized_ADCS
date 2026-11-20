from .planetary_body import PlanetaryBody


class Earth(PlanetaryBody):
    def __init__(self):
        super().__init__("Earth", 3.986004418e14, 6378137.0)
