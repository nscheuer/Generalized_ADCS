from __future__ import annotations

__all__ = ["CSatellite"]

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass, field, InitVar

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite

class CSatellite()