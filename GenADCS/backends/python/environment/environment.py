from dataclasses import dataclass, field

from .atmosphere import AtmosphericModel
from .ephemeris import EphemerisModel
from .gravity import JGravityModel
from .magnetic_field_model import MagneticFieldModel
from .planetary_body import Earth, PlanetaryBody


@dataclass(slots=True)
class Environment:
    planetary_body: PlanetaryBody = field(default_factory=Earth)
    atmosphere: AtmosphericModel | None = None
    magnetic_field: MagneticFieldModel | None = None
    j_gravity: JGravityModel | None = None
    ephemeris: EphemerisModel | None = None
