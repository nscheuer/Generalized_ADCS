__all__ = ["LiveRecorder"]

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ADCS.environment.ephemeris import DE421 as Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.animations.godot.shadow import shadow_factor
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.state import State
from ADCS.helpers.animations.godot.derive import derive_environment, j2000_to_jd_tt
from ADCS.helpers.animations.godot.from_results import manifest_metadata
from ADCS.helpers.animations.godot.scene import build_scene
from ADCS.helpers.animations.godot.schema import REQUIRED_CHANNELS, Channel, channel_spec
from ADCS.helpers.animations.godot.stream import VizServer
from ADCS.helpers.animations.godot.writer import VizLogWriter, make_manifest


class LiveRecorder:
    r"""
    Stream a running simulation to a :class:`~ADCS.helpers.animations.godot.stream.VizServer`.

    Created by :func:`~ADCS.simulate` when ``viz_stream`` is given. Sun, Moon
    and Earth orientation are derived for the whole time grid up front (one
    vectorized pass), so each step only packs a preallocated row and hands it
    to the server.

    Live rows contain the required channels plus ``r_moon``,
    ``sunlit_binary``, wheel momentum/speed and actuator commands. Torques,
    goals, estimates and sensor channels are only in the full post-run log
    (``viz_log``).

    :param server: Server to publish to. In lockstep mode the recorder waits
        up to ``wait_for_viewer`` seconds for a viewer before the first step.
    :param satellite: True satellite model (for the scene and wheel inertia).
    :param ephem: Ephemeris of the simulation.
    :param start_j2000: Start time, J2000 centuries (TT).
    :param dt: Step [s].
    :param n_steps: Number of steps.
    :param run_id: Run identifier in the stream.
    :param tee_path: Also append the streamed rows to a vizlog at this path.
    :param wait_for_viewer: Seconds to wait for a viewer in lockstep mode.
    :param bookmarks: Timeline bookmarks (:mod:`ADCS.helpers.animations.godot.timeline`), by ``t_s``.
    :param sites: Ground sites whose passes the timeline shows.
    """

    def __init__(
        self,
        server: VizServer,
        satellite: Satellite,
        ephem: Ephemeris,
        start_j2000: float,
        dt: float,
        n_steps: int,
        run_id: str = "0",
        tee_path: Optional[str | Path] = None,
        wait_for_viewer: Optional[float] = 30.0,
        bookmarks: Optional[List[Any]] = None,
        sites: Optional[List[Any]] = None,
    ) -> None:
        self.server = server
        self.run_id = str(run_id)
        self.start_j2000 = float(start_j2000)
        self.dt = float(dt)

        j2000 = self.start_j2000 + np.arange(n_steps) * self.dt * TimeConstants.sec2cent
        self._env = derive_environment(ephem, j2000_to_jd_tt(j2000))
        self._rw_J = np.array([rw.J for rw in satellite.rw_actuators], dtype=float)

        channels: List[Channel] = [channel_spec(n) for n in REQUIRED_CHANNELS]
        channels += [channel_spec("r_moon"), channel_spec("sunlit_binary")]
        scene = build_scene(satellite)
        if satellite.number_RW > 0:
            rw_ids = [a["id"] for a in scene["actuators"] if a["type"] == "rw"]
            channels += [channel_spec("h_wheel", satellite.number_RW, labels=rw_ids),
                         channel_spec("wheel_speed", satellite.number_RW, labels=rw_ids),
                         channel_spec("h_rw_body")]
            self._rw_axes = np.vstack([rw.axis for rw in satellite.rw_actuators])
        if satellite.control_len > 0:
            channels.append(channel_spec("u_cmd", satellite.control_len,
                                         labels=[a["id"] for a in scene["actuators"]]))
        self.channels = channels

        offsets, i = {}, 0
        for c in channels:
            offsets[c.name] = slice(i, i + c.width)
            i += c.width
        self._slices = offsets
        self._row = np.zeros(i, dtype="<f8")
        self._last_q: Optional[np.ndarray] = None

        from ADCS.helpers.animations.godot.overlays import default_overlays, validate_overlays

        info = {c.name: c.to_json() for c in channels}
        scene["overlays"] = validate_overlays(default_overlays(info, scene), info, scene)
        if bookmarks or sites:
            from ADCS.helpers.animations.godot.timeline import validate_timeline

            scene["timeline"] = validate_timeline(bookmarks, sites, t_range=(0.0, (n_steps - 1) * self.dt))

        epoch_jd = float(j2000_to_jd_tt(self.start_j2000))
        epoch_iso = ephem.ts.tt_jd(epoch_jd).utc_iso(places=3)
        metadata = manifest_metadata(ephem)
        manifest = make_manifest(channels, epoch_jd, epoch_iso, metadata)
        manifest["runs"] = [{"id": self.run_id, "n_rows": int(n_steps), "dt_s": self.dt}]

        self._tee = None
        if tee_path is not None:
            self._tee = VizLogWriter(tee_path, channels, epoch_jd_tt=epoch_jd, epoch_iso_utc=epoch_iso,
                                     scene=scene, metadata=metadata, overwrite=True)
            self._tee.begin_run(self.run_id, dt_s=self.dt)

        server.begin_live(manifest, scene)
        if server.mode == "lockstep":
            server.wait_for_client(wait_for_viewer)

    def step(self, k: int, x: State, os: Orbital_State, u: Optional[np.ndarray]) -> None:
        r"""Publish the state at step ``k`` (the start of the step, as recorded)."""
        s, row, env = self._slices, self._row, self._env
        q = np.asarray(x.q, dtype=float)
        if self._last_q is not None and q @ self._last_q < 0.0:
            q = -q
        self._last_q = q

        row[s["t_s"]] = k * self.dt
        row[s["r_sc"]] = os.R
        row[s["v_sc"]] = os.V
        row[s["q_gcrs_body"]] = q
        row[s["w_body"]] = x.w
        row[s["r_sun"]] = env["r_sun"][k]
        row[s["q_itrs_gcrs"]] = env["q_itrs_gcrs"][k]
        row[s["shadow_factor"]] = shadow_factor(os.R, env["r_sun"][k])
        row[s["r_moon"]] = env["r_moon"][k]
        row[s["sunlit_binary"]] = float(os.is_sunlit())
        if "h_wheel" in s:
            row[s["h_wheel"]] = x.h
            row[s["wheel_speed"]] = np.asarray(x.h, dtype=float) / self._rw_J
            row[s["h_rw_body"]] = np.asarray(x.h, dtype=float) @ self._rw_axes
        if "u_cmd" in s:
            row[s["u_cmd"]] = np.nan if u is None else u

        self.server.publish_row(self.run_id, row)
        if self._tee is not None:
            self._tee.append(self.run_id, **{c.name: row[s[c.name]] for c in self.channels})

    def close(self) -> None:
        r"""End the run and the session; complete the tee log."""
        self.server.end_run(self.run_id)
        self.server.end_session()
        if self._tee is not None:
            self._tee.close()
