"""User-configurable hierarchical state machine for the OBC."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, cast


class Trigger:
    """Base class for state-machine transition triggers."""


@dataclass(frozen=True, slots=True)
class Telecommand(Trigger):
    """Transition when the named telecommand arrives, optionally overriding ADCS goal or hardware."""

    command: str
    goal: Any = None
    sensors: Any = None
    actuators: Any = None

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("telecommand name must be nonempty")


@dataclass(frozen=True, slots=True)
class InFOV(Trigger):
    """Transition when a geodetic coordinate enters (or leaves) the orbital/nadir swath cone.

    Parameters
    ----------
    lat, lon, alt:
        Target WGS84 geodetic latitude (deg), longitude (deg), altitude (km).
    max_off_nadir_deg:
        Maximum off-nadir angle (deg) defining the orbital swath cone.
    negate:
        If True, triggers when the coordinate is outside the swath cone.
    goal:
        Optional ADCS ``Goal`` to command on transition (defaults to ``CoordinateGoal(lat, lon, alt)``
        when ``negate`` is False if no mode binding overrides it).
    """

    lat: float
    lon: float
    alt: float = 0.0
    max_off_nadir_deg: float = 45.0
    negate: bool = False
    goal: Any = None

    def __post_init__(self) -> None:
        if not all(map(math.isfinite, (self.lat, self.lon, self.alt, self.max_off_nadir_deg))):
            raise ValueError("InFOV parameters must be finite")
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError("latitude must be between -90 and 90 degrees")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError("longitude must be between -180 and 180 degrees")
        if not 0.0 < self.max_off_nadir_deg <= 90.0:
            raise ValueError("max_off_nadir_deg must be in (0, 90]")

    def __invert__(self) -> InFOV:
        return InFOV(
            lat=self.lat,
            lon=self.lon,
            alt=self.alt,
            max_off_nadir_deg=self.max_off_nadir_deg,
            negate=not self.negate,
            goal=None,
        )


@dataclass(frozen=True, slots=True)
class Completion(Trigger):
    """Transition automatically when the current mode's ``duration`` elapses."""


@dataclass(frozen=True, slots=True)
class After(Trigger):
    """Transition automatically after ``duration`` seconds in the source mode."""

    duration: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("duration must be finite and positive seconds")


@dataclass(frozen=True, slots=True)
class FDIRTriggered(Trigger):
    """Transition when an FDIR monitor raises a fault (optional monitor filter)."""

    monitor: str | None = None


@dataclass(slots=True)
class Mode:
    """Operating mode (optionally containing nested submodes)."""

    name: str
    submodes: list[Mode] = field(default_factory=lambda: cast(list[Mode], []))
    initial_submode: str | None = None
    duration: float | None = None
    auto_target: str | None = None
    bindings: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or "." in self.name or "*" in self.name:
            raise ValueError(f"invalid mode name: {self.name!r}")
        if self.duration is not None and (
            not math.isfinite(self.duration) or self.duration <= 0.0
        ):
            raise ValueError("mode duration must be finite and positive seconds")
        if self.submodes:
            names = [sub.name for sub in self.submodes]
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate submode names in {self.name!r}")
            if self.initial_submode is None:
                self.initial_submode = self.submodes[0].name
            elif self.initial_submode not in names:
                raise ValueError(
                    f"initial_submode {self.initial_submode!r} not in {self.name!r}"
                )
        elif self.initial_submode is not None:
            raise ValueError("initial_submode requires submodes")


@dataclass(frozen=True, slots=True)
class Transition:
    """Directed transition between two mode paths (supports ``*`` wildcards)."""

    source: str
    target: str
    trigger: Trigger
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("transition source and target must be nonempty")
        if not isinstance(self.trigger, Trigger):
            raise TypeError(f"invalid trigger: {type(self.trigger).__name__}")


@dataclass(slots=True)
class StateMachine:
    """Mission-level OBC state machine with modes, submodes, and transitions."""

    initial: str
    modes: list[Mode]
    transitions: list[Transition] = field(
        default_factory=lambda: cast(list[Transition], [])
    )
    timeline: list[tuple[float, str] | tuple[float, str, Any]] = field(
        default_factory=lambda: cast(list[tuple[float, str] | tuple[float, str, Any]], [])
    )

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("StateMachine requires at least one Mode")
        names = [mode.name for mode in self.modes]
        if len(set(names)) != len(names):
            raise ValueError("duplicate top-level mode names")
        if not self.initial:
            raise ValueError("initial mode must be nonempty")
        for entry in self.timeline:
            if len(entry) not in (2, 3):
                raise ValueError("timeline entries must be (time, command) or (time, command, goal)")
            t, cmd = entry[0], entry[1]
            if not math.isfinite(t) or t < 0.0:
                raise ValueError("timeline times must be finite and nonnegative")
            if not cmd:
                raise ValueError("timeline telecommand names must be nonempty")
