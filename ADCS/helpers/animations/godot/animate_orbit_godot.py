"""Open the bundled Godot viewer for an ADCS orbit simulation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import numpy as np

from ADCS.helpers.simresults import RunResults, SimulationResults
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.state import State

from .from_results import results_to_vizlog
from .launcher import view


def animate_orbit_godot(
    time_hist: Sequence[float] | None = None,
    state_hist: Sequence[State] | None = None,
    os_hist: Sequence[Orbital_State] | None = None,
    est_state_hist: Sequence[State] | None = None,
    est_os_hist: Sequence[Orbital_State] | None = None,
    boresight_goal_hist: np.ndarray | None = None,
    coord_goal=None,
    texture_path: str | Path | None = None,
    *,
    satellite: Satellite | None = None,
    results: SimulationResults | RunResults | None = None,
    model: str | Path | None = None,
    ephemeris=None,
    graphics_quality: str = "low",
    output: str | Path | None = None,
    binary: str | Path | None = None,
    assets: str | Path | None = None,
    launch: bool = True,
) -> int:
    """Write simulation frames and open the Godot viewer.

    The first seven positional parameters mirror ``animate_orbit_panda3d``.
    ``results`` preserves all recorded channels and is preferred when available.
    The generated vizlog is kept when ``output`` is provided. Graphics assets
    are local only; this function never downloads them.
    """
    if graphics_quality not in {"low", "medium", "high"}:
        raise ValueError("graphics_quality must be low, medium, or high")
    if coord_goal is not None:
        raise ValueError("coord_goal is not a recorded history; provide a target history instead")
    if texture_path is not None:
        raise ValueError("texture_path is unsupported; prepare local Godot assets and pass assets=...")
    if results is not None and any(x is not None for x in (time_hist, state_hist, os_hist)):
        raise ValueError("pass results or histories, not both")
    if results is None:
        if any(x is None for x in (time_hist, state_hist, os_hist)):
            raise ValueError("time_hist, state_hist, and os_hist are required")
        if not (len(time_hist) == len(state_hist) == len(os_hist)) or not len(time_hist):
            raise ValueError("histories must have the same nonzero length")
        if satellite is None:
            satellite = Satellite(mass=1.0, J_0=np.eye(3))
        run = RunResults(
            satellite=satellite,
            time_s=np.asarray(time_hist, dtype=float),
            time_J2000=np.asarray([os.J2000 for os in os_hist], dtype=float),
            state_hist=list(state_hist), os_hist=list(os_hist),
            est_state_hist=list(est_state_hist) if est_state_hist is not None else None,
            est_os_hist=list(est_os_hist) if est_os_hist is not None else None,
            target_hist=np.asarray(boresight_goal_hist) if boresight_goal_hist is not None else None,
        )
        results = SimulationResults([run])
    elif isinstance(results, RunResults):
        results = SimulationResults([results])

    extra_args = ("--quality", {"low": "fast", "medium": "balanced", "high": "high"}[graphics_quality])
    if output is not None:
        log = results_to_vizlog(results, output, ephem=ephemeris, model=model, overwrite=True, torques="none")
        return view(log, binary=binary, assets=assets, extra_args=extra_args, launch=launch)
    with TemporaryDirectory(prefix="adcs-godot-") as tmp:
        log = results_to_vizlog(results, Path(tmp) / "simulation.vizlog", ephem=ephemeris, model=model, torques="none")
        return view(log, binary=binary, assets=assets, extra_args=extra_args, launch=launch)
