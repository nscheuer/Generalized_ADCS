"""Star catalogues and sky-region queries for attitude sensors."""

from .catalogue import NavigationStar, StarCatalog
from .hipparcos import HipparcosCatalog
from .bright_star import BrightStarCatalog

StarCatalogue = StarCatalog
StarCatalogue.Hipparcos = HipparcosCatalog
StarCatalogue.BrightStar = BrightStarCatalog

__all__ = ["NavigationStar", "StarCatalog", "StarCatalogue", "HipparcosCatalog", "BrightStarCatalog"]

