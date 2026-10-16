from .magnetic_field import (
    coefficients_at,
    igrf_gc,
    DEFAULT_MAGNETIC_MODEL,
    MAGNETIC_MODELS,
    MagneticField,
    MagneticFieldModel,
    IGRF14,
    IGRF14Numba,
    IGRF14Ppigrf,
    WMM2025,
    WMM,
    CHAOS8,
    field_gc,
)
from .star_catalogue import NavigationStar, StarCatalog, StarCatalogue, HipparcosCatalog, BrightStarCatalog

__all__ = [
    "NavigationStar",
    "StarCatalog",
    "StarCatalogue",
    "HipparcosCatalog",
    "BrightStarCatalog",
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
    "WMM2025",
    "WMM",
    "CHAOS8",
]


def __getattr__(name):
    r"""Resolve environment classes lazily to avoid orbit import cycles.

    :param name: Requested exported class name.
    :type name: str
    :return: Atmosphere, ephemeris, or environment class.
    :rtype: type
    :raises AttributeError: If the name is not a lazy export.
    """
    # These model wrappers import orbit implementations. Keep them lazy because
    # Orbital_State itself imports ADCS.environment while ADCS.orbits loads.
    if name in {"Atmosphere", "SMADAtmosphere", "NRLMSISE00", "NRLMSISE", "AtmosphericModel"}:
        from .atmosphere import Atmosphere, SMADAtmosphere, NRLMSISE00, NRLMSISE, AtmosphericModel
        return {"Atmosphere": Atmosphere, "SMADAtmosphere": SMADAtmosphere, "NRLMSISE00": NRLMSISE00, "NRLMSISE": NRLMSISE, "AtmosphericModel": AtmosphericModel}[name]
    if name in {"Ephemeris", "DE421", "DE442", "DE442s", "DE442S", "DE441", "VSOP2013", "EphemerisModel"}:
        from .ephemeris import Ephemeris, DE421, DE442, DE442s, DE442S, DE441, VSOP2013, EphemerisModel
        return {"Ephemeris": Ephemeris, "DE421": DE421, "DE442": DE442, "DE442s": DE442s, "DE442S": DE442S, "DE441": DE441, "VSOP2013": VSOP2013, "EphemerisModel": EphemerisModel}[name]
    if name == "Environment":
        from .environment import Environment
        return Environment
    raise AttributeError(name)


__all__ += ["Atmosphere", "SMADAtmosphere", "NRLMSISE00", "NRLMSISE", "AtmosphericModel", "Ephemeris", "DE421", "DE442", "DE442s", "DE442S", "DE441", "VSOP2013", "EphemerisModel", "Environment"]
