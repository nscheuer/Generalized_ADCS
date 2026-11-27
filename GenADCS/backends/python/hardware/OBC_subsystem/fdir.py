"""Fault Detection, Isolation, and Recovery (FDIR) monitors for the OBC."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import cast


class FDIRMonitor:
    """Base class for compiled FDIR monitors."""

    name: str


@dataclass(frozen=True, slots=True)
class AngularRateMonitor(FDIRMonitor):
    """Trigger an FDIR fault when body angular rate magnitude exceeds ``max_rate`` (rad/s)."""

    max_rate: float
    name: str = "angular_rate"
    target_mode: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_rate) or self.max_rate <= 0.0:
            raise ValueError("max_rate must be finite and positive rad/s")
        if not self.name:
            raise ValueError("monitor name must be nonempty")


@dataclass(slots=True)
class FDIR:
    """FDIR supervisor running on the OBC to detect faults and command safe mode."""

    monitors: list[FDIRMonitor] = field(
        default_factory=lambda: cast(list[FDIRMonitor], [])
    )
    safe_mode: str = "safe_mode"

    def __post_init__(self) -> None:
        if not self.safe_mode:
            raise ValueError("safe_mode must be a nonempty mode name")
        seen: set[str] = set()
        for monitor in self.monitors:
            if not isinstance(monitor, FDIRMonitor):
                raise TypeError(f"unsupported FDIR monitor: {type(monitor).__name__}")
            if monitor.name in seen:
                raise ValueError(f"duplicate FDIR monitor name: {monitor.name!r}")
            seen.add(monitor.name)
