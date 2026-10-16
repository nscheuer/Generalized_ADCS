"""ppigrf-backed IGRF-14 compatibility model."""

from .igrf14 import IGRF14


class IGRF14Ppigrf(IGRF14):
    """IGRF-14 evaluated by the optional ppigrf compatibility implementation."""

    def __init__(self) -> None:
        super().__init__("igrf_ppigrf")
