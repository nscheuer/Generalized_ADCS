"""Numba-backed IGRF-14 model."""

from .igrf14 import IGRF14


class IGRF14Numba(IGRF14):
    """IGRF-14 evaluated by the in-tree Numba implementation."""

    def __init__(self) -> None:
        super().__init__("igrf_numba")
