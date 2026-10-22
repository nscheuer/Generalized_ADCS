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
        self.focus_radius = focus_radius
        self.distance = focus_radius * 5.2
        self.min_distance = focus_radius * 1.12
        self.max_distance = 1.0e11
        self.azimuth = 0.0
        self.elevation = math.atan2(0.65, 5.2)
        self.dragging = False
        self._last_mouse = None
        self._last_focus_position = None
        self._apply()

    def set_focus(self, node, *, radius: float, distance: float | None = None,
                  preserve_distance: bool = False) -> None:
        """Select the node that the camera orbits; it may move during playback."""
        current_distance = self.distance
        self.focus = node
        self.focus_radius = radius
        self.min_distance = radius * 1.12
        self.max_distance = 1.0e11
        if preserve_distance:
            self.distance = current_distance
            # Switching focus must not change the requested zoom, even when
            # the new target is physically larger than the current distance.
            self.min_distance = min(self.min_distance, self.distance)
        else:
            self.distance = distance if distance is not None else radius * 5.2
            self.distance = max(self.min_distance, min(self.max_distance, self.distance))
        self._last_focus_position = None
        self._apply()

    def update_aspect(self, window, reserved_bottom: float) -> None:
        if window is not None and window.getYSize() > 0:
            # Use the drawable viewport's pixel dimensions rather than
            # inferring them from the window and reserved UI height.  Panda
            # rounds normalized display-region edges to pixels, so the
            # inferred ratio can drift during interactive window resizing.
            display_region = self.base.camNode.getDisplayRegion(0)
            width = display_region.getPixelWidth()
            height = display_region.getPixelHeight()
            if width > 0 and height > 0:
                self.base.camLens.setAspectRatio(width / height)
            else:
                self.base.camLens.setAspectRatio(
                    window.getXSize() /
                    (window.getYSize() * (1.0 - reserved_bottom))
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
        step = 1.5 if self.distance > max(self.focus_radius * 10, 1.0e5) else 1.14
        multiplier = 1.0 / step if direction < 0 else step
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
        # Keep depth precision stable over the range from a planetary close-up
        # to a view wider than Neptune's orbit.
        self.base.camLens.setNearFar(max(1.0, self.distance * 0.001),
                                     self.distance * 100.0)
        self._last_focus_position = focus_position
