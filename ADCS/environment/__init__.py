from .igrf import coefficients_at, igrf_gc
from .magnetic_field import (
    DEFAULT_MAGNETIC_MODEL,
    MAGNETIC_MODELS,
    MagneticField,
    MagneticFieldModel,
    IGRF14,
    IGRF14Numba,
    IGRF14Ppigrf,
    field_gc,
)
from .star_catalogue import NavigationStar, StarCatalog, StarCatalogue, HipparcosCatalog

__all__ = [
    "NavigationStar",
    "StarCatalog",
    "StarCatalogue",
    "HipparcosCatalog",
    "coefficients_at",
    "igrf_gc",
    "DEFAULT_MAGNETIC_MODEL",
    "MAGNETIC_MODELS",
    "field_gc",
    "MagneticFieldModel",
    "MagneticField",
    "IGRF14",
    "IGRF14Numba",
    "IGRF14Ppigrf",
]


def __getattr__(name):
    # These model wrappers import orbit implementations. Keep them lazy because
    # Orbital_State itself imports ADCS.environment while ADCS.orbits loads.
    if name in {"Atmosphere", "SMADAtmosphere", "AtmosphericModel"}:
        from .atmosphere import Atmosphere, SMADAtmosphere, AtmosphericModel
        return {"Atmosphere": Atmosphere, "SMADAtmosphere": SMADAtmosphere, "AtmosphericModel": AtmosphericModel}[name]
    if name in {"Ephemeris", "DE421", "DE442", "DE441", "VSOP2013", "EphemerisModel"}:
        from .ephemeris import Ephemeris, DE421, DE442, DE441, VSOP2013, EphemerisModel
        return {"Ephemeris": Ephemeris, "DE421": DE421, "DE442": DE442, "DE441": DE441, "VSOP2013": VSOP2013, "EphemerisModel": EphemerisModel}[name]
    if name == "Environment":
        from .environment import Environment
        return Environment
    raise AttributeError(name)


__all__ += ["Atmosphere", "SMADAtmosphere", "AtmosphericModel", "Ephemeris", "DE421", "DE442", "DE441", "VSOP2013", "EphemerisModel", "Environment"]
