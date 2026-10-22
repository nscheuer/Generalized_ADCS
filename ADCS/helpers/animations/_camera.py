"""Camera controls and viewport projection, independent of scene content."""

import math


class OrbitCamera:
    """Orbit a focus node using absolute pointer motion and bounded wheel zoom.

    The focus can later be any scene node. Distance limits belong to the focus
    object, rather than to the input handler, so a new target can supply its
    own useful scale without changing the camera controls.
    """

    def __init__(self, base, focus, focus_radius: float) -> None:
        self.base = base
        self.focus = focus
        self.distance = focus_radius * 5.2
        self.min_distance = focus_radius * 1.12
        self.max_distance = focus_radius * 30.0
        self.azimuth = 0.0
        self.elevation = math.atan2(0.65, 5.2)
        self.dragging = False
        self._last_mouse = None
        self._last_focus_position = None
        self._apply()

    def set_focus(self, node, *, radius: float, distance: float | None = None) -> None:
        """Select the node that the camera orbits; it may move during playback."""
        self.focus = node
        self.min_distance = radius * 1.12
        self.max_distance = radius * 30.0
        self.distance = distance if distance is not None else radius * 5.2
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))
        self._last_focus_position = None
        self._apply()

    def update_aspect(self, window, reserved_bottom: float) -> None:
        if window is not None and window.getYSize() > 0:
            self.base.camLens.setAspectRatio(
                window.getXSize() / (window.getYSize() * (1.0 - reserved_bottom))
            )

    def begin_drag(self, mouse) -> None:
        self.dragging = True
        self._last_mouse = (float(mouse.x), float(mouse.y))

    def end_drag(self) -> None:
        self.dragging = False
        self._last_mouse = None

    def drag_to(self, mouse) -> None:
        if not self.dragging:
            return
        current = (float(mouse.x), float(mouse.y))
        if self._last_mouse is None:
            self._last_mouse = current
            return
        dx = current[0] - self._last_mouse[0]
        dy = current[1] - self._last_mouse[1]
        self._last_mouse = current
        # Screen movement right rotates the view left around the focus.
        self.azimuth -= dx * 2.8
        self.elevation = max(-1.48, min(1.48, self.elevation - dy * 2.2))
        self._apply()

    def zoom(self, direction: int) -> None:
        multiplier = 0.88 if direction < 0 else 1.14
        self.distance = max(
            self.min_distance, min(self.max_distance, self.distance * multiplier)
        )
        self._apply()

    def update_focus(self) -> None:
        """Keep a moving focus centered, without resetting the view angles."""
        position = self.focus.getPos(self.base.render)
        if self._last_focus_position is None or position != self._last_focus_position:
            self._apply()

    def _apply(self) -> None:
        focus_position = self.focus.getPos(self.base.render)
        horizontal = self.distance * math.cos(self.elevation)
        self.base.camera.setPos(
            focus_position.x + horizontal * math.sin(self.azimuth),
            focus_position.y - horizontal * math.cos(self.azimuth),
            focus_position.z + self.distance * math.sin(self.elevation),
        )
        self.base.camera.lookAt(focus_position)
        self._last_focus_position = focus_position
