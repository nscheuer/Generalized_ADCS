"""Visualization asset pipeline: raw public datasets -> viewer-ready cube maps and tables.

Run ``python -m ADCS.viz.assets --help``. Needs Pillow (``pip install "generalized-adcs[viz-assets]"``).
"""

from .fetch import cache_dir, fetch, load_sources
from .products import (
           PRODUCTS,
           build,
           current_products,
           verify_processed,
           write_attribution,
)

__all__ = ["PRODUCTS", "build", "cache_dir", "current_products", "fetch", "load_sources",
           "verify_processed", "write_attribution"]
