from .igrf import coefficients_at, igrf_gc
from .magnetic_field import DEFAULT_MAGNETIC_MODEL, MAGNETIC_MODELS, field_gc
from .star_catalog import NavigationStar, StarCatalog

__all__ = [
    "NavigationStar",
    "StarCatalog",
    "coefficients_at",
    "igrf_gc",
    "DEFAULT_MAGNETIC_MODEL",
    "MAGNETIC_MODELS",
    "field_gc",
]
