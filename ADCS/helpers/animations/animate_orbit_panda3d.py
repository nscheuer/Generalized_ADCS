"""Panda3D orbit viewer prototype and PyVista-compatible entry point.

The viewer coordinates independent playback, camera, scene, and UI modules.
It renders Earth and an ephemeris-driven solar system using kilometres.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.state import State

from ._camera import OrbitCamera
from ._context_menu import BodyContextMenu
from ._history import PlaybackClock
from ._loading_overlay import LoadingOverlay
from ._runtime import configure_runtime
from ._scene import EarthBody, SceneGraph
from ._timeline import TimelinePanel
from ._timeline_model import TimelineConfig, TimelineInterval, TimelineModel, TimelineTrack
from ._performance import PerformanceOverlay
from ._settings_panel import SettingsPanel
from ._solar_system import SolarSystemLayer
from ._satellite_model import SatelliteModel, model_path, MODEL_URLS
from ._skybox import Starfield
from ._texture_assets import ensure_textures, tier_textures

__all__ = ["OrbitPanda3D", "animate_orbit_panda3d"]


class OrbitPanda3D:
    """Interactive viewer for recorded ADCS states and solar-system ephemerides.

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
        ephemeris=None,
        graphics_quality: str | None = None,
        satellite_model: str = '1ru',
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

        self.graphics_quality = graphics_quality or 'low'
        if self.graphics_quality not in ('low', 'medium', 'high'):
            raise ValueError("graphics_quality must be 'low', 'medium', or 'high'")
        satellite_model = satellite_model.lower()
        if satellite_model not in MODEL_URLS:
            raise ValueError(f"satellite_model must be one of {', '.join(MODEL_URLS)}")
        configure_runtime(loadPrcFileData)
        self.base = ShowBase(windowType=window_type)
        self._report_graphics_device()
        self._window_type = window_type
        self._closed = False
        # ShowBase normally calls sys.exit() when its window closes. A library
        # viewer should stop its own loop and return control to its caller.
        self.base.finalizeExit = self._request_shutdown
        self.base.disableMouse()
        self.base.setBackgroundColor(0, 0, 0, 1)
        self.base.camNode.getDisplayRegion(0).setDimensions(0, 1, self.timeline_config.height_fraction, 1)
        self.base.camLens.setNearFar(1, 1e7)
        self.base.camLens.setFov(42)

        loading = LoadingOverlay(self.base)
        try:
            ensure_textures(
                tier_textures(self.graphics_quality),
                on_progress=lambda done, total, name: loading.update(
                    0.7 * done / max(total, 1), name if name.startswith('Check terminal:') else f'Downloading {name}'
                ),
            )
            loading.update(0.72, 'Loading star map')
            self.starfield = Starfield(self.base, self.graphics_quality)
            loading.update(0.82, 'Loading Earth')
            earth_radius = float(EarthConstants.R_e)
            self.earth_body = EarthBody(self.base, earth_radius, texture_path, self.graphics_quality)
            self.earth = self.earth_body.node
            if ephemeris is None and has_any_core_history and os_hist:
                ephemeris = os_hist[0].ephem
            loading.update(0.90, 'Loading solar system')
            self.solar_system = SolarSystemLayer(self.base, self.earth_body, ephemeris, self.graphics_quality)
            self.solar_system.update(self.playback.current, self.playback.speed)
            self.world_root = self.base.render.attachNewNode('solar scene origin')
            for body in self.solar_system.bodies.values():
                body.root.wrtReparentTo(self.world_root)
            for trail in self.solar_system._orbit_trails.values():
                trail.render = self.world_root
            self.satellite = None
            if has_any_core_history and len(self.time):
                loading.update(0.93, 'Loading CubeSat model')
                path = model_path(satellite_model, on_progress=lambda done, total, label:
                                  loading.update(0.93 + 0.05 * done / max(total, 1), label))
                self.satellite = SatelliteModel(self.base, path, self.time, state_hist,
                                                os_hist, self.earth_body.root)
            loading.update(1.0, 'Ready')
        finally:
            loading.close()
        self.scene = SceneGraph()
        self.camera = OrbitCamera(self.base, self.earth_body.root, earth_radius)
        self._focus_name = 'earth'
        if self.satellite is not None:
            self._install_spacecraft_camera()
            self.satellite.update(self.playback.simulation_time(self.playback.current),
                                  self.earth_body.root.getPos(self.world_root),
                                  self.base.camera.getPos(self.base.render),
                                  center_on_satellite=True)
            position = self.satellite.absolute_position
            self.world_root.setPos(*-position)
            self.camera.set_focus(self.satellite.root, radius=self.satellite.radius_km)
            self._focus_name = 'satellite'
            self.satellite.update_marker()
        self.starfield.update(self.base.camera, self.camera.distance)
        self.timeline = TimelinePanel(
            self.base, self.playback, self.timeline_model, self.timeline_config,
            on_reset_view=self.reset_view, on_reset_time=self.reset_time,
        )
        self.solar_system.set_label_font(self.timeline.font)
        self.performance = PerformanceOverlay(self.base, self.timeline.font)
        self.settings = SettingsPanel(self.base, self.timeline.font)
        for tier in ('low', 'medium', 'high'):
            self.settings.add_toggle(
                f'graphics_{tier}', f'{tier.title()} graphics',
                lambda tier=tier: self.graphics_quality == tier,
                lambda selected, tier=tier: self.set_graphics_quality(tier) if selected else None,
            )
        self.context_menu = BodyContextMenu(
            self.base, self.timeline.font,
            lambda name: self.focus_satellite() if name == 'satellite'
            else self.focus_body(name, preserve_zoom=True),
        )
        self._pending_body_hit = None
        self._pointer_start = None
        self._menu_pointer_active = False
        self._settings_pointer_active = False
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode
        self.focus_label = OnscreenText(
            text=f'FOCUS: {self._focus_name.upper()}', pos=(0, 0.96), scale=0.026,
            align=TextNode.ALeft, fg=(0.75, 0.88, 0.98, 1),
            font=self.timeline.font, mayChange=True, parent=self.base.aspect2d,
        )
        self._layout_focus_label()
        self._on_window_event(self.base.win)

        self._rendered_time = None
        self._render_time()
        self.base.accept("window-event", self._on_window_event)
        self.base.accept("space", self.playback.toggle)
        self.base.accept("mouse1", self._start_pointer_action)
        self.base.accept("mouse1-up", self._end_pointer_action)
        self.base.accept("mouse3", self._open_context_menu)
        self.base.accept("escape", self._close_context_menu)
        self.base.accept("wheel_up", self._zoom, [-1])
        self.base.accept("wheel_down", self._zoom, [1])
        self.base.accept("page_up", self.timeline.scroll_tracks, [-1])
        self.base.accept("page_down", self.timeline.scroll_tracks, [1])
        self.base.accept("tab", self._cycle_focus)
        self.base.accept("s", self.focus_body, ["sun"])
        self.base.accept("e", self.focus_body, ["earth"])
        if self.satellite is not None:
            self.base.accept('c', self.focus_satellite)
        self.base.taskMgr.add(self._update, "orbit viewer update")

    def _report_graphics_device(self) -> None:
        """Print the renderer and GPU selected by Panda3D's graphics backend."""
        gsg = self.base.win.getGsg() if self.base.win is not None else None
        pipe_name = self.base.pipe.getInterfaceName() if self.base.pipe else 'unknown'
        if gsg is None:
            print(f'Panda3D graphics: {pipe_name}; graphics device unavailable')
            return
        renderer = gsg.getDriverRenderer() or 'unknown renderer'
        vendor = gsg.getDriverVendor() or 'unknown vendor'
        version = gsg.getDriverVersion() or 'unknown driver version'
        renderer_lower = renderer.lower()
        if any(token in renderer_lower for token in ('llvmpipe', 'softpipe', 'software rasterizer')):
            device = f'{renderer} (software renderer; no GPU model is active)'
        else:
            # Hardware OpenGL drivers normally include the adapter model in
            # GL_RENDERER (for example, "AMD Radeon ..." or "NVIDIA ...").
            device = renderer
        print(f'Panda3D GPU/iGPU: {device}')
        print(f'Panda3D graphics driver: {vendor} / OpenGL {version}')

    @property
    def elapsed(self) -> float:
        return self.playback.current - self.playback.start_utc.timestamp()

    @property
    def playing(self) -> bool:
        return self.playback.playing

    @playing.setter
    def playing(self, value: bool) -> None:
        self.playback.playing = bool(value)

    def _install_spacecraft_camera(self) -> None:
        """Render metre-scale geometry with its own depth range."""
        from panda3d.core import BitMask32, Camera, PerspectiveLens

        planetary_mask = BitMask32.bit(1)
        spacecraft_mask = BitMask32.bit(2)
        self.base.camNode.setCameraMask(planetary_mask)
        self.satellite.root.hide(planetary_mask)
        self.starfield.root.hide(spacecraft_mask)
        for body in self.solar_system.bodies.values():
            body.root.hide(spacecraft_mask)
        self.spacecraft_lens = PerspectiveLens()
        self.spacecraft_lens.setFov(self.base.camLens.getHfov(), self.base.camLens.getVfov())
        self.spacecraft_lens.setAspectRatio(self.base.camLens.getAspectRatio())
        self.spacecraft_camera = self.base.camera.attachNewNode(Camera('spacecraft camera'))
        self.spacecraft_camera.node().setLens(self.spacecraft_lens)
        self.spacecraft_camera.node().setCameraMask(spacecraft_mask)
        self.spacecraft_region = self.base.win.makeDisplayRegion(
            0, 1, self.timeline_config.height_fraction, 1)
        self.spacecraft_region.setCamera(self.spacecraft_camera)
        self.spacecraft_region.setSort(1)
        self.spacecraft_region.setClearDepthActive(True)
        self.spacecraft_region.setClearColorActive(False)
        self.camera.near_lens = self.spacecraft_lens
        self.camera._apply()

    def _on_window_event(self, window) -> None:
        if window is not None:
            # Keep Panda's 2D coordinate bounds in sync before laying out HUD
            # and timeline nodes; the scene lens then uses its shorter viewport.
            self.base.adjustWindowAspectRatio(self.base.getAspectRatio(window))
        self.camera.update_aspect(window, self.timeline_config.height_fraction)
        if window is not None and window.getYSize() > 0:
            self.timeline.resize()
            self.performance.resize()
            self.settings.resize()
            self._layout_focus_label()
            self.solar_system.resize_ui()
            self.solar_system.update_lod()
            self.context_menu.resize()

    def _layout_focus_label(self) -> None:
        """Keep the focus indicator inside the current aspect2d screen bounds."""
        left, right = self.base.a2dLeft, self.base.a2dRight
        bottom, top = self.base.a2dBottom, self.base.a2dTop
        ui_scale = max(1.0, top)
        margin_x = min(0.07 * ui_scale, (right - left) * 0.04)
        margin_top = min(0.04 * ui_scale, (top - bottom) * 0.025)
        available_width = max(0.1, right - left - 2 * margin_x)
        text_width = max(self.focus_label.textNode.getWidth(), 1e-6)
        scale = min(0.026 * ui_scale, available_width / text_width)
        self.focus_label.setScale(scale)
        self.focus_label.setPos(left + margin_x, top - margin_top - scale)

    def _render_time(self) -> None:
        current = self.playback.current
        if current == self._rendered_time:
            return
        self.scene.update(self.playback.simulation_time(current))
        self.solar_system.update(current, self.playback.speed)
        self.timeline.set_time(current)
        self._rendered_time = current

    def focus_body(self, name: str, *, preserve_zoom: bool = False) -> None:
        """Center and orbit any body while retaining physical scene scale."""
        self.world_root.setPos(0, 0, 0)
        if self.satellite is not None:
            self.satellite.root.setPos(*self.satellite.absolute_position)
        body = self.solar_system.set_focus(name)
        self.camera.set_focus(body.root, radius=body.spec.radius_km,
                              preserve_distance=preserve_zoom)
        self._focus_name = name
        self.focus_label.setText(f'FOCUS: {name.upper()}')
        self._layout_focus_label()
        self.solar_system.update_lod()
        if self.satellite is not None:
            self.satellite.update_marker()

    def focus_satellite(self) -> None:
        if self.satellite is None:
            return
        position = self.satellite.absolute_position
        self.world_root.setPos(*-position)
        self.satellite.root.setPos(0, 0, 0)
        self.camera.set_focus(self.satellite.root, radius=self.satellite.radius_km)
        self._focus_name = 'satellite'
        self.satellite.update_marker()
        self.focus_label.setText('FOCUS: SPACECRAFT')
        self._layout_focus_label()

    def set_graphics_quality(self, tier: str) -> None:
        if tier == self.graphics_quality:
            return
        if tier not in ('low', 'medium', 'high'):
            raise ValueError("tier must be 'low', 'medium', or 'high'")
        loading = LoadingOverlay(self.base)
        try:
            ensure_textures(
                tier_textures(tier),
                on_progress=lambda done, total, name: loading.update(
                    0.7 * done / max(total, 1), name if name.startswith('Check terminal:') else f'Downloading {name}'
                ),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f'Graphics quality unchanged: {exc}')
            loading.close()
            return
        try:
            loading.update(0.75, 'Loading star map')
            self.starfield.set_quality(tier)
            loading.update(0.85, 'Loading planet textures')
            self.earth_body.set_graphics_quality(tier)
            self.solar_system.set_quality(tier)
            self.graphics_quality = tier
            loading.update(0.95, 'Updating scene')
            self.solar_system.update_lod()
            loading.update(1.0, 'Ready')
        finally:
            loading.close()

    def reset_view(self) -> None:
        """Restore the initial spacecraft or Earth focus and orbit angles."""
        if self.satellite is not None:
            self.focus_satellite()
        else:
            self.focus_body("earth")
        self.camera.azimuth = 0.0
        self.camera.elevation = float(np.arctan2(0.65, 5.2))
        self.camera._apply()

    def reset_time(self) -> None:
        """Return the viewer to its initial UTC and pause there."""
        self.playback.seek(self.playback.start_utc.timestamp())
        self.playback.playing = False
        self.playback.speed = 1.0
        self.timeline.follow = True
        self.timeline.center = self.playback.current + self.timeline.span * 0.25
        self.timeline._rebuild_ruler()
        self._rendered_time = None
        self._render_time()

    def _cycle_focus(self) -> None:
        names = (('satellite',) if self.satellite is not None else ()) + self.solar_system.FOCUS_ORDER
        index = (names.index(self._focus_name) + 1) % len(names)
        if names[index] == 'satellite':
            self.focus_satellite()
        else:
            self.focus_body(names[index])

    def _pick_focus_target(self, mouse):
        if self.satellite is not None and self.satellite.pick(mouse):
            return 'satellite'
        return self.solar_system.pick_body(mouse)

    def _start_pointer_action(self) -> None:
        watcher = self.base.mouseWatcherNode
        if watcher is None or not watcher.hasMouse():
            return
        mouse = watcher.getMouse()
        if self.settings.handle_pointer(mouse):
            self._settings_pointer_active = True
            return
        if self.context_menu.visible:
            if self.context_menu.contains(mouse):
                self._menu_pointer_active = True
                return
            self.context_menu.hide()
        if self.timeline.contains(mouse):
            self.timeline.begin_pointer(mouse)
            self._render_time()
        else:
            self._pointer_start = (float(mouse.x), float(mouse.y))
            self._pending_body_hit = self._pick_focus_target(mouse)
            self.camera.begin_drag(mouse)

    def _end_pointer_action(self) -> None:
        if self._settings_pointer_active:
            self._settings_pointer_active = False
            return
        watcher = self.base.mouseWatcherNode
        mouse = watcher.getMouse() if watcher is not None and watcher.hasMouse() else None
        if self._menu_pointer_active:
            if not self.context_menu.activate(mouse):
                self.context_menu.hide()
            self._menu_pointer_active = False
            return
        self.timeline.end_pointer(mouse)
        self.camera.end_drag()
        if self._pending_body_hit is not None and mouse is not None and self._pointer_start is not None:
            dx = float(mouse.x) - self._pointer_start[0]
            dy = float(mouse.y) - self._pointer_start[1]
            if dx * dx + dy * dy <= 0.0004:
                self.context_menu.show(self._pending_body_hit, mouse)
        self._pending_body_hit = None
        self._pointer_start = None

    def _open_context_menu(self) -> None:
        watcher = self.base.mouseWatcherNode
        if watcher is None or not watcher.hasMouse():
            return
        mouse = watcher.getMouse()
        if self.timeline.contains(mouse):
            self.context_menu.hide()
            return
        name = self._pick_focus_target(mouse)
        if name is None:
            self.context_menu.hide()
        else:
            self.context_menu.show(name, mouse)

    def _close_context_menu(self) -> None:
        self.context_menu.hide()

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
        self.earth_body.update_camera()
        self.starfield.update(self.base.camera, self.camera.distance)
        frame_dt = self.base.clock.getDt()
        self.playback.advance(frame_dt)
        self.performance.update(frame_dt)
        self._render_time()
        if self.satellite is not None:
            self.satellite.update(
                self.playback.simulation_time(self.playback.current),
                self.earth_body.root.getPos(self.world_root),
                self.base.camera.getPos(self.base.render),
                center_on_satellite=self._focus_name == 'satellite',
            )
            if self._focus_name == 'satellite':
                position = self.satellite.absolute_position
                self.world_root.setPos(*-position)
            self.camera.update_focus()
            self.satellite.update_marker()
        self.solar_system.update_lod()
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
        self.focus_label.destroy()
        self.context_menu.close()
        self.solar_system.close()
        if self.satellite is not None:
            self.satellite.close()
            self.base.win.removeDisplayRegion(self.spacecraft_region)
            self.spacecraft_camera.removeNode()
        self.earth_body.close()
        self.starfield.close()
        self.world_root.removeNode()
        self.performance.destroy()
        self.settings.close()
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
    ephemeris=None,
    graphics_quality: str | None = None,
    satellite_model: str = '1ru',
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
    Scroll over the scene to zoom from Earth to the solar system. S focuses
    the Sun, E returns to Earth, and Tab cycles through bodies. The solar
    scene uses the first recorded orbit's ephemeris when available, otherwise
    the bundled DE442s ephemeris; pass ``ephemeris`` to select one explicitly.
    ``graphics_quality`` accepts ``'low'``, ``'medium'``, or ``'high'`` and
    defaults to ``'low'``.
    With simulation histories, ``satellite_model`` selects ``'1ru'``,
    ``'2ru'``, ``'icecube'``, or ``'mirata'``. Press C to focus it.
    """
    OrbitPanda3D(
        time_hist, state_hist, os_hist, est_state_hist, est_os_hist,
        boresight_goal_hist, coord_goal, texture_path,
        start_utc=start_utc, timeline_config=timeline_config,
        timeline_model=timeline_model, ephemeris=ephemeris,
        graphics_quality=graphics_quality,
        satellite_model=satellite_model,
    ).run()
