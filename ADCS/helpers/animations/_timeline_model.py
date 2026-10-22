"""Timeline data independent of Panda3D widgets or simulation playback."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite


def utc_seconds(value: datetime | float) -> float:
    """Normalize a UTC datetime or Unix timestamp to seconds."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    result = float(value)
    if not isfinite(result):
        raise ValueError("Timeline times must be finite")
    return result


@dataclass(frozen=True)
class TimelineInterval:
    start: datetime | float
    end: datetime | float
    label: str = ""
    color: tuple[float, float, float, float] = (0.15, 0.72, 0.91, 1.0)

    def __post_init__(self) -> None:
        if utc_seconds(self.end) < utc_seconds(self.start):
            raise ValueError("Timeline interval end precedes start")


@dataclass
class TimelineTrack:
    key: str
    label: str
    intervals: list[TimelineInterval] = field(default_factory=list)
    children: list['TimelineTrack'] = field(default_factory=list)

    def add_child(self, track: 'TimelineTrack') -> 'TimelineTrack':
        self.children.append(track)
        return track


@dataclass
class TimelineModel:
    tracks: list[TimelineTrack] = field(default_factory=list)

    def add_track(self, track: TimelineTrack) -> TimelineTrack:
        self.tracks.append(track)
        return track

    def rows(self):
        """Yield (depth, track) in display order; nested tracks become subtracks."""
        def walk(track, depth):
            yield depth, track
            for child in track.children:
                yield from walk(child, depth + 1)
        for track in self.tracks:
            yield from walk(track, 0)


@dataclass(frozen=True)
class TimelineConfig:
    height_fraction: float = 0.26
    initial_span_seconds: float | None = None
    min_span_seconds: float = 0.01
    max_span_seconds: float = 100 * 365.25 * 86400
    row_height: float = 0.053
    zoom_factor: float = 1.5

    def __post_init__(self) -> None:
        if not 0.17 <= self.height_fraction <= 0.45:
            raise ValueError("height_fraction must be between 0.17 and 0.45")
        if self.min_span_seconds <= 0 or self.max_span_seconds <= self.min_span_seconds:
            raise ValueError("Invalid timeline span limits")
        if self.initial_span_seconds is not None and self.initial_span_seconds <= 0:
            raise ValueError("initial_span_seconds must be positive")
        if self.row_height <= 0 or self.zoom_factor <= 1:
            raise ValueError("Invalid timeline layout or zoom factor")
