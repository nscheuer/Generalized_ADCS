"""Ephemeris sampled past-orbit arcs for currently labeled bodies."""

import time

import numpy as np
from skyfield.errors import EphemerisRangeError


# Sidereal orbital periods in days; the Moon's period is relative to Earth.
ORBIT_PERIOD_DAYS = {
    'mercury': 87.9691,
    'venus': 224.701,
    'earth': 365.256,
    'moon': 27.321661,
    'mars': 686.980,
    'jupiter': 4332.59,
    'saturn': 10759.22,
    'uranus': 30688.5,
    'neptune': 60182.0,
}


class OrbitTrail:
    """Draw the last 45 degrees of a body's ephemeris path."""

    REFRESH_SECONDS = 1.0
    SAMPLE_COUNT = 257
    SWEEP_RADIANS = np.deg2rad(45.0)

    def __init__(self, render, color) -> None:
        self.render = render
        self.color = color
        self.node = None
        self._last_update_wall = -float('inf')
        self._last_utc = None
        self._last_focus = None

    def hide(self) -> None:
        if self.node is not None:
            self.node.hide()

    def update(self, ephemeris_track, name: str, utc: float,
               current_positions: dict[str, np.ndarray], focus_name: str) -> None:
        now = time.monotonic()
        if self._last_focus is not None and self._last_focus != focus_name:
            self._last_update_wall = -float('inf')
        if (self.node is not None and self._last_utc == utc
                and self._last_focus == focus_name):
            self.node.show()
            return
        if (self.node is not None and self._last_utc is not None
                and now - self._last_update_wall < self.REFRESH_SECONDS):
            self.node.show()
            return
        if now - self._last_update_wall < self.REFRESH_SECONDS:
            return

        try:
            points = self._sample_points(
                ephemeris_track, name, utc, current_positions, focus_name
            )
        except (ValueError, OverflowError, OSError, EphemerisRangeError):
            self.hide()
            self._last_update_wall = now
            return
        if points is None or len(points) < 2:
            self.hide()
            self._last_update_wall = now
            return

        from panda3d.core import LineSegs, OmniBoundingVolume, TransparencyAttrib
        segments = LineSegs(f'{name} past 45 degree orbit trail')
        segments.setThickness(1.35)
        segments.setColor(self.color[0], self.color[1], self.color[2], 0.68)
        segments.moveTo(*points[0])
        for point in points[1:]:
            segments.drawTo(*point)
        node = self.render.attachNewNode(segments.create())
        node.setDepthTest(True)
        node.setDepthWrite(False)
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setBin('transparent', 1)
        # Relative planet coordinates can be billions of kilometres from the
        # viewer origin; finite local bounds lose precision at those scales.
        node.node().setBounds(OmniBoundingVolume())
        node.node().setFinal(True)
        if self.node is not None:
            self.node.removeNode()
        self.node = node
        self._last_update_wall = now
        self._last_utc = utc
        self._last_focus = focus_name

    def _sample_points(self, track, name, utc, current_positions, focus_name):
        period = ORBIT_PERIOD_DAYS.get(name)
        if period is None:
            return None
        start = utc - period * 86400.0 / 2.0
        sample_times = np.linspace(start, utc, self.SAMPLE_COUNT)
        samples = track.orbit_positions_at_utc(name, sample_times)
        if name == 'moon':
            # The Moon's path is Earth-relative. Shift it into the current
            # viewer origin, matching the live Moon position.
            center_offset = (current_positions['earth']
                             - current_positions[focus_name])
            current_vector = current_positions['moon']
            orbit_samples = samples
            orbit_vector = current_vector
        else:
            # Planet paths are heliocentric. Place the Sun at its current
            # inertial scene position, then shift by the selected focus.
            center_offset = (current_positions['sun']
                             - current_positions[focus_name])
            current_vector = current_positions[name] - current_positions['sun']
            orbit_samples = samples
            orbit_vector = current_vector

        current_norm = np.linalg.norm(orbit_vector)
        sample_norms = np.linalg.norm(orbit_samples, axis=1)
        if current_norm <= 0 or np.any(sample_norms <= 0):
            return None
        cos_angles = (orbit_samples @ orbit_vector) / (sample_norms * current_norm)
        angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
        # Samples are chronological, so the angular distance generally falls
        # from the older half-orbit toward zero. Select the last crossing into
        # the 45-degree window to keep only the recent arc.
        crossings = np.flatnonzero(
            (angles[:-1] >= self.SWEEP_RADIANS)
            & (angles[1:] < self.SWEEP_RADIANS)
        )
        if crossings.size == 0:
            return None
        index = int(crossings[-1])
        older_angle, newer_angle = angles[index], angles[index + 1]
        fraction = ((older_angle - self.SWEEP_RADIANS)
                    / (older_angle - newer_angle))
        first = (orbit_samples[index]
                 + fraction * (orbit_samples[index + 1] - orbit_samples[index]))
        orbit_path = np.vstack((first, orbit_samples[index + 1:], orbit_vector))
        return orbit_path + center_offset
