"""Star catalogues and sky-region queries for attitude sensors."""

from .catalogue import NavigationStar, StarCatalog
from .hipparcos import HipparcosCatalog

StarCatalogue = StarCatalog
StarCatalogue.Hipparcos = HipparcosCatalog

__all__ = ["NavigationStar", "StarCatalog", "StarCatalogue", "HipparcosCatalog"]

