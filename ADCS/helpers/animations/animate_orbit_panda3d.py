"""Panda3D orbit viewer prototype and PyVista-compatible entry point.

The viewer coordinates independent playback, camera, scene, and UI modules.
The current scene contains Earth only; all distances use kilometres.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.state import State

from ._camera import OrbitCamera
from ._history import EarthOrientationTrack, PlaybackClock
from ._runtime import configure_runtime
from ._scene import EarthBody, EarthLayer, SceneGraph
from ._timeline import TimelinePanel
from ._timeline_model import TimelineConfig, TimelineInterval, TimelineModel, TimelineTrack
from ._performance import PerformanceOverlay

__all__ = ["OrbitPanda3D", "animate_orbit_panda3d"]


class OrbitPanda3D:
    """Interactive viewer for recorded ADCS states and orbital states.

Additional scene bodies can be added alongside ``EarthBody``. Camera focus
is managed by ``OrbitCamera`` and can follow a scene node. Expensive orbital
frame conversion is completed before the render loop starts.
    """

    def __init__(
        self,
        time_hist: Optional[np.ndarray] = None,
        state_hist: Optional[Sequence[State]] = None,
        os_hist: Optional[Sequence[Orbital_State]] = None,
        est_state_hist: Optional[Sequence[State]] = None,
        est_os_hist: Optional[Sequence[Orbital_State]] = None,
        boresight_goal_hist: Optional[np.ndarray] = None,
        coord_goal=None,
        texture_path: Optional[str | Path] = None,
        *,
        window_type: str = "onscreen",
        start_utc: datetime | None = None,
        timeline_config: TimelineConfig | None = None,
        timeline_model: TimelineModel | None = None,
    ) -> None:
        # The extra histories preserve the established PyVista call signature.
        # Scene layers may consume them as they are implemented.
        core_histories = (time_hist, state_hist, os_hist)
        has_any_core_history = any(history is not None for history in core_histories)
        if has_any_core_history and not all(history is not None for history in core_histories):
            raise ValueError("time_hist, state_hist, and os_hist must be provided together")
        if not has_any_core_history and any(
            history is not None
            for history in (est_state_hist, est_os_hist, boresight_goal_hist, coord_goal)
        ):
            raise ValueError("simulation detail histories require time_hist, state_hist, and os_hist")

        if has_any_core_history:
            if start_utc is None and len(os_hist):
                start_utc = os_hist[0].datetime
            self.playback = PlaybackClock(time_hist, start_utc)
            if len(self.playback.time) != len(os_hist):
                raise ValueError("time_hist and os_hist must have the same length")
            if len(self.playback.time) != len(state_hist):
                raise ValueError("time_hist and state_hist must have the same length")
        else:
            self.playback = PlaybackClock(None, start_utc)
            os_hist = ()
        self.orientation = EarthOrientationTrack(self.playback.time, os_hist)
        self.time = self.playback.time
        self.timeline_config = timeline_config or TimelineConfig()
        self.timeline_model = timeline_model if timeline_model is not None else TimelineModel()
        if has_any_core_history and timeline_model is None and self.playback.data_interval:
            start, end = self.playback.data_interval
            self.timeline_model.add_track(TimelineTrack(
                'simulation', 'Simulation', [TimelineInterval(start, end)]
            ))

        from direct.showbase.ShowBase import ShowBase
        from panda3d.core import loadPrcFileData

        configure_runtime(loadPrcFileData)
        self.base = ShowBase(windowType=window_type)
        self._window_type = window_type
        self._closed = False
        # ShowBase normally calls sys.exit() when its window closes. A library
        # viewer should stop its own loop and return control to its caller.
        self.base.finalizeExit = self._request_shutdown
        self.base.disableMouse()
        self.base.setBackgroundColor(0.008, 0.015, 0.035, 1)
        self.base.camNode.getDisplayRegion(0).setDimensions(0, 1, self.timeline_config.height_fraction, 1)
        self.base.camLens.setNearFar(1, 1e7)
        self.base.camLens.setFov(42)

        earth_radius = float(EarthConstants.R_e)
        self.earth_body = EarthBody(self.base, earth_radius, texture_path)
        self.earth = self.earth_body.node
        self.scene = SceneGraph()
        self.scene.add("earth", EarthLayer(self.earth_body, self.orientation))
        self.camera = OrbitCamera(self.base, self.earth_body.root, earth_radius)
        self.timeline = TimelinePanel(self.base, self.playback, self.timeline_model,
                                      self.timeline_config)
        self.performance = PerformanceOverlay(self.base, self.timeline.font)
        self._on_window_event(self.base.win)

        self._rendered_time = None
        self._render_time()
        self.base.accept("window-event", self._on_window_event)
        self.base.accept("space", self.playback.toggle)
        self.base.accept("mouse1", self._start_pointer_action)
        self.base.accept("mouse1-up", self._end_pointer_action)
        self.base.accept("wheel_up", self._zoom, [-1])
        self.base.accept("wheel_down", self._zoom, [1])
        self.base.accept("page_up", self.timeline.scroll_tracks, [-1])
        self.base.accept("page_down", self.timeline.scroll_tracks, [1])
        self.base.taskMgr.add(self._update, "orbit viewer update")

    @property
    def elapsed(self) -> float:
        return self.playback.current - self.playback.start_utc.timestamp()

    @property
    def playing(self) -> bool:
        return self.playback.playing

    @playing.setter
    def playing(self, value: bool) -> None:
        self.playback.playing = bool(value)

    def _on_window_event(self, window) -> None:
        self.camera.update_aspect(window, self.timeline_config.height_fraction)
        if window is not None and window.getYSize() > 0:
            self.timeline.resize()
            self.performance.resize()

    def _render_time(self) -> None:
        current = self.playback.current
        if current == self._rendered_time:
            return
        self.scene.update(self.playback.simulation_time(current))
        self.timeline.set_time(current)
        self._rendered_time = current

    def _start_pointer_action(self) -> None:
        watcher = self.base.mouseWatcherNode
        if watcher is None or not watcher.hasMouse():
            return
        mouse = watcher.getMouse()
        if self.timeline.contains(mouse):
            self.timeline.begin_pointer(mouse)
            self._render_time()
        else:
            self.camera.begin_drag(mouse)

    def _end_pointer_action(self) -> None:
        watcher = self.base.mouseWatcherNode
        self.timeline.end_pointer(watcher.getMouse() if watcher is not None and watcher.hasMouse() else None)
        self.camera.end_drag()

    def _zoom(self, direction: int) -> None:
        watcher = self.base.mouseWatcherNode
        if watcher is not None and watcher.hasMouse() and self.timeline.contains(watcher.getMouse()):
            self.timeline.zoom(watcher.getMouse(), direction)
            return
        self.camera.zoom(direction)

    def _update(self, task):
        if self._window_type == "onscreen":
            window = self.base.win
            if (
                window is None
                or not window.isValid()
                or not window.getProperties().getOpen()
            ):
                self._request_shutdown()
                return task.done

        watcher = self.base.mouseWatcherNode
        if watcher is not None and watcher.hasMouse():
            mouse = watcher.getMouse()
            if self.timeline.interacting:
                self.timeline.move_pointer(mouse)
            if self.camera.dragging:
                self.camera.drag_to(mouse)

        self.camera.update_focus()
        frame_dt = self.base.clock.getDt()
        self.playback.advance(frame_dt)
        self.performance.update(frame_dt)
        self._render_time()
        return task.cont

    def run(self) -> None:
        """Run until the window closes, then release Panda resources."""
        try:
            self.base.run()
        finally:
            self.close()

    def _request_shutdown(self) -> None:
        self.base.taskMgr.stop()

    def close(self) -> None:
        """Close the viewer and free its window, tasks, and graphics resources."""
        if self._closed:
            return
        self._closed = True
        self.base.taskMgr.stop()
        self.base.taskMgr.remove("orbit viewer update")
        self.performance.destroy()
        self.base.destroy()


def animate_orbit_panda3d(
    time_hist: Optional[np.ndarray] = None,
    state_hist: Optional[Sequence[State]] = None,
    os_hist: Optional[Sequence[Orbital_State]] = None,
    est_state_hist: Optional[Sequence[State]] = None,
    est_os_hist: Optional[Sequence[Orbital_State]] = None,
    boresight_goal_hist: Optional[np.ndarray] = None,
    coord_goal=None,
    texture_path: Optional[str | Path] = None,
    *,
    start_utc: datetime | None = None,
    timeline_config: TimelineConfig | None = None,
    timeline_model: TimelineModel | None = None,
) -> None:
    """Open the viewer, optionally using the histories from an ADCS simulation.

    Call with no arguments to open the Earth viewer without satellite playback.
    When histories are supplied, provide ``time_hist``, ``state_hist``, and
    ``os_hist`` together. The UTC clock starts at the first orbital state's
    timestamp unless ``start_utc`` is provided. Pass ``timeline_model`` to
    display custom tracks and nested subsystem tracks, and ``timeline_config``
    to set the panel height, initial time span, and zoom limits. Drag the
    ruler to pan, click it to seek, and use the wheel over it to zoom. Page
    Up/Down scroll track rows; the playback buttons or Space control time.
    """
    OrbitPanda3D(
        time_hist, state_hist, os_hist, est_state_hist, est_os_hist,
        boresight_goal_hist, coord_goal, texture_path,
        start_utc=start_utc, timeline_config=timeline_config,
        timeline_model=timeline_model,
    ).run()
