__all__ = ["results_to_vizlog"]

import copy
import platform
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ADCS.helpers.math_helpers import angle_between_vectors, quaternion_angle, rot_mat
from ADCS.helpers.simresults import RunResults, SimulationResults
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.shadow import shadow_factor
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.satellite_hardware.errors import ErrorMode
from ADCS.viz.derive import derive_environment, j2000_to_jd_tt
from ADCS.viz.interp import quat_continuity
from ADCS.viz.scene import build_scene, sensor_layout
from ADCS.viz.schema import REQUIRED_CHANNELS, TARGET_KIND, Channel, channel_spec
from ADCS.viz.writer import VizLogWriter

#: Disturbance/actuator torques recomputed with bias but no noise, and
#: without advancing any random state (see results_to_vizlog).
_NOMINAL_MODE = ErrorMode(add_bias=True, add_noise=False, update_bias=False, update_noise=False)


def _generator() -> Dict[str, Any]:
    import ADCS

    sha = None
    try:
        repo = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=False)
        if out.returncode == 0:
            sha = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {"name": "Generalized_ADCS", "version": getattr(ADCS, "__version__", None),
            "git_sha": sha, "python": platform.python_version()}


def manifest_metadata(ephem: Ephemeris) -> Dict[str, Any]:
    r"""Manifest entries describing the producer, ephemeris and body models."""
    return {
        "generator": _generator(),
        "ephemeris": ephem.provenance(),
        "earth": {"model": "WGS84", "a_km": 6378.137, "f": 1.0 / 298.257223563,
                  "R_e_sim_km": EarthConstants.R_e},
        "bodies": {"sun_radius_km": EarthConstants.R_sun, "moon_radius_km": EarthConstants.R_moon},
        "shadow_model": "conical (Montenbruck & Gill 3.4.2), occulter radius R_e_sim_km",
    }


def _orbital_states(run: RunResults, ephem: Ephemeris) -> List[Orbital_State]:
    hist = run.os_hist or []
    return [os if isinstance(os, Orbital_State) else Orbital_State.from_dict(os, ephem=ephem) for os in hist]


def _target_columns(target_hist: Any, n: int, R: np.ndarray, q: np.ndarray,
                    boresight: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    kind = np.zeros((n, 1))
    target = np.full((n, 4), np.nan)
    err = np.full((n, 1), np.nan)
    raw = list(target_hist)
    for k in range(n):
        t = np.asarray(raw[k], dtype=float).reshape(-1)
        if t.size == 3:
            t = np.concatenate([[np.nan], t])
        if t.size != 4 or not np.any(np.isfinite(t)) or np.allclose(np.nan_to_num(t), 0.0):
            continue
        if np.isnan(t[0]):
            vec = t[1:]
            if np.linalg.norm(vec) > 10.0:  # a position [km], as AnimationPlot treats it
                kind[k] = TARGET_KIND["position"]
                direction = vec - R[k]
            else:
                kind[k] = TARGET_KIND["direction"]
                direction = vec
            target[k] = t
            if boresight is not None and np.all(np.isfinite(boresight[k])):
                err[k] = angle_between_vectors(rot_mat(q[k]) @ boresight[k], direction)
        else:
            kind[k] = TARGET_KIND["quaternion"]
            target[k] = t
            err[k] = quaternion_angle(q[k], t)
    return {"target_kind": kind, "target": target, "pointing_err": err}


def _nominal_torques(run: RunResults, states: List[Any], oss: List[Orbital_State],
                     controls: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    sat = copy.deepcopy(run.satellite)
    n = len(states)
    u_zero = np.zeros(sat.control_len)
    tau_a = np.empty((n, 3))
    tau_d = np.empty((n, 3))
    try:
        for k in range(n):
            u = controls[k] if controls is not None else u_zero
            tau_d[k] = sat.dist_torques(x=states[k], os=oss[k], dmode=_NOMINAL_MODE)
            tau_a[k] = sat.act_torque(x=states[k], u=u, os=oss[k], dmode=_NOMINAL_MODE)
    except Exception as e:  # noqa: BLE001 - optional channel; report and continue
        warnings.warn(f"vizlog: torque recompute failed ({type(e).__name__}: {e}); "
                      "omitting tau_act_body/tau_dist_body", stacklevel=3)
        return None
    return tau_a, tau_d


def _run_columns(run: RunResults, ephem: Ephemeris, torques: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Channel]]:
    oss = _orbital_states(run, ephem)
    states = list(run.state_hist or [])
    n = len(states)
    if n == 0 or len(oss) != n:
        raise ValueError(f"run has {n} states but {len(oss)} orbital states")
    sat = run.satellite

    j2000 = np.array(run.time_J2000 if run.time_J2000 is not None else [o.J2000 for o in oss], dtype=float)
    R = np.array([o.R for o in oss], dtype=float)
    V = np.array([o.V for o in oss], dtype=float)
    q = quat_continuity(np.array([s.q for s in states], dtype=float))
    env = derive_environment(ephem, j2000_to_jd_tt(j2000))

    cols: Dict[str, np.ndarray] = {
        "_j2000": j2000,
        # Elapsed seconds as the simulation recorded them (k * dt), which are
        # exact; differencing J2000 centuries loses ~1e-8 s.
        "_time_s": (np.array(run.time_s, dtype=float) if run.time_s is not None and len(run.time_s) == n
                    else (j2000 - j2000[0]) * TimeConstants.cent2sec),
        "r_sc": R,
        "v_sc": V,
        "q_gcrs_body": q,
        "w_body": np.array([s.w for s in states], dtype=float),
        "r_sun": env["r_sun"],
        "q_itrs_gcrs": env["q_itrs_gcrs"],
        "shadow_factor": shadow_factor(R, env["r_sun"]),
        "r_moon": env["r_moon"],
        "sunlit_binary": np.array([float(o.is_sunlit()) for o in oss]),
        "b_field": np.array([getattr(o, "B", np.full(3, np.nan)) for o in oss], dtype=float),
        "rho": np.array([getattr(o, "rho", np.nan) for o in oss], dtype=float),
    }
    specs: Dict[str, Channel] = {}

    actuator_ids = [a["id"] for a in build_scene(sat)["actuators"]]
    if sat.number_RW > 0:
        h = np.array([s.h for s in states], dtype=float).reshape(n, sat.number_RW)
        J = np.array([rw.J for rw in sat.rw_actuators], dtype=float)
        rw_ids = [a for a in actuator_ids if a.startswith("rw")]
        cols["h_wheel"], cols["wheel_speed"] = h, h / J
        specs["h_wheel"] = channel_spec("h_wheel", sat.number_RW, labels=rw_ids)
        specs["wheel_speed"] = channel_spec("wheel_speed", sat.number_RW, labels=rw_ids)

    controls = None
    if run.control_hist is not None and sat.control_len > 0:
        controls = np.array(run.control_hist, dtype=float).reshape(n, sat.control_len)
        cols["u_cmd"] = controls
        specs["u_cmd"] = channel_spec("u_cmd", sat.control_len, labels=actuator_ids)

    if torques == "nominal":
        result = _nominal_torques(run, states, oss, controls)
        if result is not None:
            cols["tau_act_body"], cols["tau_dist_body"] = result
            prov = "nominal_recompute: bias applied, noise off, random state not advanced"
            specs["tau_act_body"] = channel_spec("tau_act_body", provenance=prov)
            specs["tau_dist_body"] = channel_spec("tau_dist_body", provenance=prov)
    elif torques != "none":
        raise ValueError(f"torques must be 'nominal' or 'none', got {torques!r}")

    boresight = None
    if run.boresight_hist is not None and len(run.boresight_hist) == n:
        boresight = np.array([np.asarray(b, dtype=float).reshape(3) for b in run.boresight_hist])
        cols["boresight_body"] = boresight
    if run.target_hist is not None and len(run.target_hist) == n:
        cols.update(_target_columns(run.target_hist, n, R, q, boresight))
    if run.w_target_hist is not None and len(run.w_target_hist) == n:
        cols["w_target"] = np.array([np.asarray(w, dtype=float).reshape(3) for w in run.w_target_hist])

    est = run.est_state_hist
    if est is not None and len(est) == n and all(e is not None for e in est):
        cols["q_est"] = quat_continuity(np.array([e.q for e in est], dtype=float))
        cols["w_est"] = np.array([e.w for e in est], dtype=float)
    est_os = run.est_os_hist
    if est_os is not None and len(est_os) == n and all(e is not None for e in est_os):
        def _rv(e: Any, key: str) -> np.ndarray:
            e = getattr(e, "os", e)
            return np.asarray(e[key] if isinstance(e, dict) else getattr(e, key), dtype=float)
        cols["r_est"] = np.array([_rv(e, "R") for e in est_os])
        cols["v_est"] = np.array([_rv(e, "V") for e in est_os])

    if run.sensor_hist is not None and len(run.sensor_hist) == n:
        y = np.array([np.asarray(v, dtype=float).reshape(-1) for v in run.sensor_hist])
        layout = sensor_layout(sat)
        if layout and layout[-1]["slice"][1] == y.shape[1]:
            for block in layout:
                i0, i1 = block["slice"]
                name = f"sensor_{block['id']}"
                cols[name] = y[:, i0:i1]
                specs[name] = channel_spec(name, i1 - i0, unit="native",
                                           description=f"{block['type']} reading (noisy)")
        else:
            warnings.warn("vizlog: sensor_hist width does not match the sensor layout; "
                          "omitting sensor channels", stacklevel=3)

    for name in cols:
        if not name.startswith("_") and name not in specs:
            specs[name] = channel_spec(name)
    return cols, specs


def results_to_vizlog(
    sim: SimulationResults,
    path: str | Path,
    ephem: Optional[Ephemeris] = None,
    torques: str = "nominal",
    overwrite: bool = False,
) -> Path:
    r"""
    Write simulation results as an ``adcs-vizlog`` directory.

    This is a post-processing pass: it never runs inside the simulation loop.
    Environment quantities that the results do not keep (Moon, Earth
    orientation, shadow factor) are recomputed in one vectorized pass from the
    recorded times.

    Torques (``torques="nominal"``) are recomputed on a copy of the satellite
    with bias applied but noise disabled, so they are the nominal torques at
    each recorded state, not bit-identical to what the integrator applied when
    noise is enabled. The channels carry that note as ``provenance``.

    All runs share one epoch (the first run's first sample); ``t_s`` of later
    runs is relative to it.

    :param sim: Results from :func:`~ADCS.simulate` or :func:`~ADCS.mc.simulate_mc`.
    :param path: Output directory.
    :param ephem: Ephemeris for recomputed quantities and for rebuilding
        orbital states stored as dictionaries. Defaults to the one the
        recorded orbital states use, else a new :class:`Ephemeris`.
    :param torques: ``"nominal"`` or ``"none"``.
    :param overwrite: Replace an existing vizlog at ``path``.
    :return: The vizlog path.
    """
    if ephem is None:
        first = (sim.runs[0].os_hist or [None])[0]
        ephem = getattr(first, "ephem", None) or Ephemeris()

    per_run = [_run_columns(run, ephem, torques) for run in sim.runs]

    # The channel set is that of the first run; later runs must agree on widths.
    specs = per_run[0][1]
    for i, (_, s) in enumerate(per_run[1:], start=1):
        for name, spec in specs.items():
            if name in s and s[name].width != spec.width:
                raise ValueError(f"run {i}: channel {name!r} width {s[name].width} != {spec.width}")
    required = [n for n in REQUIRED_CHANNELS if n != "t_s"]
    ordered = required + [n for n in specs if n not in REQUIRED_CHANNELS]
    channels = [channel_spec("t_s")] + [specs[n] for n in ordered]
    names = {c.name for c in channels}

    epoch_j2000 = float(per_run[0][0]["_j2000"][0])
    epoch_jd = float(j2000_to_jd_tt(epoch_j2000))
    epoch_iso = ephem.ts.tt_jd(epoch_jd).utc_iso(places=3)

    with VizLogWriter(path, channels, epoch_jd_tt=epoch_jd, epoch_iso_utc=epoch_iso,
                      scene=build_scene(sim.runs[0].satellite), metadata=manifest_metadata(ephem),
                      overwrite=overwrite) as writer:
        run_ids = sim.run_ids or list(range(len(sim.runs)))
        for rid, (cols, _) in zip(run_ids, per_run):
            j2000 = cols.pop("_j2000")
            elapsed = cols.pop("_time_s")
            cols["t_s"] = (j2000[0] - epoch_j2000) * TimeConstants.cent2sec + (elapsed - elapsed[0])
            dt = float(np.median(np.diff(cols["t_s"]))) if len(j2000) > 1 else None
            writer.write_run(rid, {k: v for k, v in cols.items() if k in names}, dt_s=dt)
    return Path(path)
