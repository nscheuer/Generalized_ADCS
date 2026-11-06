__all__ = ["build_scene", "sensor_layout"]

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.helpers.animations.godot.fov import sensor_fov


def _ids(objs: List[Any]) -> List[str]:
    counts: Dict[str, int] = {}
    out = []
    for o in objs:
        base = "rw" if isinstance(o, RW) else type(o).__name__.lower()
        out.append(f"{base}{counts.get(base, 0)}")
        counts[base] = counts.get(base, 0) + 1
    return out


def sensor_layout(satellite: Satellite) -> List[Dict[str, Any]]:
    r"""
    Where each attitude sensor's values sit in the flat ``sensor_hist`` vector.

    ``Satellite.sensor_readings`` concatenates the attitude (non-GPS) sensors in
    order, then one momentum reading per reaction wheel.

    :return: One entry per block: ``{"id", "type", "slice": [i0, i1]}``; wheel
        momentum blocks have type ``"RW_momentum"``.
    """
    layout = []
    i = 0
    for sid, s in zip(_ids(satellite.attitude_sensors), satellite.attitude_sensors):
        n = int(s.output_length)
        layout.append({"id": sid, "type": type(s).__name__, "slice": [i, i + n]})
        i += n
    for wid in _ids(satellite.rw_actuators):
        layout.append({"id": f"{wid}_h", "type": "RW_momentum", "slice": [i, i + 1]})
        i += 1
    return layout


def build_scene(satellite: Satellite, model: Optional[str | Path] = None) -> Dict[str, Any]:
    r"""
    Static description of a spacecraft for ``scene.json``.

    Body-frame geometry the viewer needs to draw the spacecraft and, later,
    its actuators and sensor fields of view. All vectors are in the BODY
    frame, lengths in metres.

    :param satellite: The (true) satellite model.
    :param model: Optional ``.viz.json`` sidecar from
        :func:`~ADCS.helpers.animations.godot.mesh_import.import_spacecraft`. The scene then points
        to the imported model and links actuators and sensors to its nodes.
    :return: JSON-serializable dictionary (numpy arrays allowed; the writer
        converts them).
    """
    actuators = []
    for idx, (aid, a) in enumerate(zip(_ids(satellite.actuators), satellite.actuators)):
        entry = {
            "id": aid,
            "index": idx,
            "type": "rw" if isinstance(a, RW) else "mtq",
            "axis_body": np.asarray(a.axis, dtype=float),
            "u_max": float(a.u_max),
            "node": None,
        }
        if isinstance(a, RW):
            entry["J_kgm2"] = float(a.J)
            entry["h_max_Nms"] = np.asarray(a.h_max, dtype=float)
        actuators.append(entry)

    sensors = []
    for block, s in zip(sensor_layout(satellite), satellite.attitude_sensors):
        entry = dict(block)
        fov = sensor_fov(s)
        if fov is not None:
            entry.update({
                "boresight_body": fov.boresight_body,
                "fov_half_rad": fov.fov_half_rad,
                "fov_assumed": fov.assumed,
                "bidirectional": fov.bidirectional,
            })
        entry["node"] = None
        sensors.append(entry)

    panels: List[Dict[str, Any]] = []
    seen: Dict[tuple, Dict[str, Any]] = {}
    for d in satellite.disturbances:
        config = getattr(d, "config", None)
        for p in getattr(config, "params", []) or []:
            key = (round(p["area"], 12), *np.round(p["centroid"], 12), *np.round(p["normal"], 12))
            if key in seen:
                seen[key]["used_by"].append(type(d).__name__)
                continue
            panel = {
                "area_m2": float(p["area"]),
                "center_body_m": np.asarray(p["centroid"], dtype=float),
                "normal_body": np.asarray(p["normal"], dtype=float),
                "eta_s": p["eta_s"], "eta_d": p["eta_d"], "eta_a": p["eta_a"], "CD": p["CD"],
                "used_by": [type(d).__name__],
            }
            seen[key] = panel
            panels.append(panel)

    mesh = None
    if model is not None:
        side_path = Path(model).resolve()
        side = json.loads(side_path.read_text(encoding="utf-8"))
        if side.get("format") != "adcs-spacecraft-model":
            raise ValueError(f"{side_path} is not a spacecraft model sidecar")
        mesh = {
            "sidecar": str(side_path),
            "sha256": hashlib.sha256(side_path.read_bytes()).hexdigest(),
            "coords": side["coords"], "units": side["units"],
            "lods": [{**lod, "path": str(side_path.parent / lod["file"])} for lod in side["lods"]],
            "nodes": {n: {k: v for k, v in e.items() if k in ("role", "actuator", "sensor", "articulation", "axis_body",
                                                               "pivot_body")}
                      for n, e in side["nodes"].items()},
        }
        for node, entry in side["nodes"].items():
            for item in actuators if "actuator" in entry else sensors if "sensor" in entry else []:
                if item["id"] == entry.get("actuator", entry.get("sensor")):
                    item["node"] = node

    return {
        "spacecraft": {
            "mass_kg": float(satellite.mass),
            "com_body_m": np.asarray(satellite.COM, dtype=float),
            "inertia_ref_kgm2": np.asarray(satellite.J_0, dtype=float),
            "inertia_com_kgm2": np.asarray(satellite.J_COM, dtype=float),
        },
        "mesh": mesh,
        "panels": panels,
        "actuators": actuators,
        "sensors": sensors,
        "boresights": {k: np.asarray(v, dtype=float) for k, v in satellite.boresight.items()},
        "disturbances": [type(d).__name__ for d in satellite.disturbances],
    }
