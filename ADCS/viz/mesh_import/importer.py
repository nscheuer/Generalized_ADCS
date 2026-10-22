r"""
Import a team's spacecraft model (glTF preferred, STL accepted) for the viewer.

The model is converted to the spacecraft BODY frame in metres, checked against
the simulation's satellite (mass properties, wheel axes, sensor boresights,
drag/SRP panels), reduced to levels of detail, and written as normalised
``.glb`` files plus a ``.viz.json`` sidecar that the vizlog scene points to.

Problems that make the model unusable are **errors**; disagreements with the
simulation model are **warnings**, because CAD usually omits internal mass and
small parts.
"""

__all__ = ["ImportResult", "import_spacecraft", "PRESET_ROTATIONS", "UNIT_SCALE"]

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

#: Model units -> metres.
UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}

#: Named CAD -> BODY rotations. "gltf_y_up": undo the Z-up -> Y-up conversion
#: Blender and most CAD exporters apply when writing glTF.
PRESET_ROTATIONS = {
    "identity": np.eye(3),
    "gltf_y_up": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
}

ROLES = ("body", "rw", "mtq", "panel", "sensor", "other")
_COMPRESSION_EXTENSIONS = ("KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_mesh_quantization")

# Thresholds (warnings).
COM_TOL_M = 0.01
COM_TOL_FRAC_DIAG = 0.02
INERTIA_REL_TOL = 0.10
PRINCIPAL_AXIS_TOL_DEG = 5.0
RW_AXIS_TOL_DEG = 2.0
BORESIGHT_TOL_DEG = 1.0
PANEL_AREA_TOL = 0.10
DEFAULT_LODS = (1.0, 0.25, 0.05)
DEFAULT_MAX_TRIANGLES = 500_000


@dataclass
class ImportResult:
    sidecar: Optional[Path]
    lods: List[Path]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def exit_code(self, strict: bool = False) -> int:
        return 2 if self.errors else (1 if strict and self.warnings else 0)


class _Abort(Exception):
    pass


# ------------------------------------------------------------------ helpers
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gltf_extensions(path: Path) -> List[str]:
    """extensionsUsed + extensionsRequired of a .glb/.gltf, without trimesh."""
    if path.suffix.lower() == ".gltf":
        doc = json.loads(path.read_text(encoding="utf-8"))
    else:
        with open(path, "rb") as f:
            magic, _version, _length = struct.unpack("<4sII", f.read(12))
            if magic != b"glTF":
                raise _Abort(f"{path.name} is not a binary glTF file")
            chunk_len, chunk_type = struct.unpack("<I4s", f.read(8))
            if chunk_type != b"JSON":
                raise _Abort(f"{path.name}: first glTF chunk is not JSON")
            doc = json.loads(f.read(chunk_len))
    return sorted(set(doc.get("extensionsUsed", [])) | set(doc.get("extensionsRequired", [])))


def _rigid(m: np.ndarray, what: str) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    if m.shape != (3, 3) or not np.allclose(m @ m.T, np.eye(3), atol=1e-6) or np.linalg.det(m) < 0:
        raise _Abort(f"{what} must be a proper rotation (orthonormal, det +1)")
    return m


def _cad_to_body(spec: Any) -> np.ndarray:
    """4x4 CAD (after unit scaling) -> BODY transform from the config."""
    out = np.eye(4)
    if spec is None:
        return out
    if isinstance(spec, list):
        a = np.asarray(spec, dtype=float)
        if a.shape != (4, 4) or not np.allclose(a[3], [0, 0, 0, 1]):
            raise _Abort("cad_to_body as a matrix must be 4x4 with last row [0, 0, 0, 1]")
        out[:3, :3] = _rigid(a[:3, :3], "cad_to_body rotation")
        out[:3, 3] = a[:3, 3]
        return out
    rot = spec.get("rotation", "identity")
    if isinstance(rot, str):
        if rot not in PRESET_ROTATIONS:
            raise _Abort(f"unknown cad_to_body rotation preset {rot!r}; choose from {list(PRESET_ROTATIONS)}")
        rot = PRESET_ROTATIONS[rot]
    out[:3, :3] = _rigid(rot, "cad_to_body rotation")
    out[:3, 3] = np.asarray(spec.get("translation_m", [0.0, 0.0, 0.0]), dtype=float)
    return out


def _node_scale(T: np.ndarray, node: str) -> float:
    """Uniform scale of a node transform; non-uniform scale or shear is an error."""
    m = T[:3, :3]
    s = np.linalg.norm(m, axis=0)
    if s.max() / s.min() > 1.0 + 1e-6:
        raise _Abort(f"node {node!r} has non-uniform scale {np.round(s, 6).tolist()}; apply scale in the CAD export")
    r = m / s
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6):
        raise _Abort(f"node {node!r} has a sheared transform; re-export without shear")
    if np.linalg.det(r) < 0:
        raise _Abort(f"node {node!r} transform mirrors geometry (det < 0)")
    return float(s.mean())


def _angle_deg(a: np.ndarray, b: np.ndarray, sign_free: bool = True) -> float:
    c = abs(a @ b) if sign_free else a @ b
    return float(np.degrees(np.arccos(np.clip(c / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def _parallel_axis(inertia_com: np.ndarray, mass: float, d: np.ndarray) -> np.ndarray:
    return inertia_com + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def _symmetry_axis(inertia: np.ndarray) -> np.ndarray:
    """Principal axis whose moment differs most from the other two."""
    w, v = np.linalg.eigh(inertia)
    k = int(np.argmax([abs(w[i] - np.mean(np.delete(w, i))) for i in range(3)]))
    return v[:, k]


# ------------------------------------------------------------------ main entry
def import_spacecraft(
    config: Dict[str, Any] | str | Path,
    out_dir: str | Path,
    satellite: Any = None,
    base_dir: Optional[Path] = None,
) -> ImportResult:
    r"""
    Import, check and normalise a spacecraft model.

    :param config: ``spacecraft_viz.json`` path or its content. Keys:
        ``name``, ``mesh`` (path relative to the config), ``units`` (required
        for STL), ``cad_to_body`` (4x4 matrix or ``{rotation: preset|3x3,
        translation_m}``), ``mass_kg`` / ``density_kg_m3`` (default: the
        satellite's mass), ``nodes`` (``{node: {role, actuator, sensor,
        articulation, axis_body, boresight_local}}``), ``lods``,
        ``max_triangles``.
    :param out_dir: Output directory for ``<name>_lod<i>.glb`` and ``<name>.viz.json``.
    :param satellite: The simulation's :class:`~ADCS.satellite_hardware.satellite.Satellite`
        to check against (optional but recommended).
    :param base_dir: Directory relative mesh paths resolve against (default:
        the config file's directory).
    :return: :class:`ImportResult` with errors, warnings and output paths.
    """
    import trimesh

    result = ImportResult(sidecar=None, lods=[])
    try:
        if isinstance(config, (str, Path)):
            base_dir = base_dir or Path(config).resolve().parent
            config = json.loads(Path(config).read_text(encoding="utf-8"))
        base_dir = base_dir or Path.cwd()
        name = str(config.get("name", "spacecraft"))
        mesh_path = (base_dir / config["mesh"]).resolve()
        if not mesh_path.exists():
            raise _Abort(f"mesh file not found: {mesh_path}")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ext = mesh_path.suffix.lower()

        # 1. Load ---------------------------------------------------------
        if ext in (".glb", ".gltf"):
            compressed = [e for e in _gltf_extensions(mesh_path) if e in _COMPRESSION_EXTENSIONS]
            if compressed:
                raise _Abort(f"compressed glTF ({', '.join(compressed)}) is not supported; "
                             "re-export without mesh compression")
            default_units = "m"
        elif ext in (".stl", ".obj", ".ply"):
            default_units = None
        else:
            raise _Abort(f"unsupported model format {ext}; use .glb/.gltf (preferred) or .stl")
        units = config.get("units", default_units)
        if units is None:
            raise _Abort(f"{ext} files carry no units; set \"units\" (one of {list(UNIT_SCALE)})")
        if units not in UNIT_SCALE:
            raise _Abort(f"unknown units {units!r}; choose from {list(UNIT_SCALE)}")
        scene = trimesh.load_scene(mesh_path)
        nodes = list(scene.graph.nodes_geometry)
        if not nodes:
            raise _Abort("model contains no geometry")

        # 2-3. Units and CAD -> BODY ------------------------------------
        scale = UNIT_SCALE[units]
        c2b = _cad_to_body(config.get("cad_to_body"))
        meshes: Dict[str, Any] = {}
        for node in nodes:
            T, geom = scene.graph[node]
            s = _node_scale(T, node)
            if abs(s - 1.0) > 1e-9:
                result.info.append(f"node {node!r} carries a uniform scale of {s:g} (applied)")
            m = scene.geometry[geom].copy()
            S = np.eye(4)
            S[:3, :3] *= scale
            m.apply_transform(c2b @ S @ T)
            meshes[node] = m
        all_v = np.vstack([m.vertices for m in meshes.values()])
        extent = all_v.max(0) - all_v.min(0)
        diag = float(np.linalg.norm(extent))
        result.info.append(f"{len(nodes)} nodes, {sum(len(m.faces) for m in meshes.values())} triangles, "
                           f"extent {np.round(extent, 4).tolist()} m (BODY)")

        node_cfg: Dict[str, Dict[str, Any]] = config.get("nodes", {})
        for n in node_cfg:
            if n not in meshes:
                result.errors.append(f"config names node {n!r}, which is not in the model; nodes: {nodes}")
        for n, c in node_cfg.items():
            if c.get("role", "body") not in ROLES:
                result.errors.append(f"node {n!r}: role must be one of {ROLES}")
        if result.errors:
            raise _Abort(result.errors.pop())

        # 4. Mesh validity --------------------------------------------------
        closed: Dict[str, bool] = {}
        for n, m in meshes.items():
            if not m.is_volume:
                trimesh.repair.fix_normals(m)
            closed[n] = bool(m.is_volume)
            if not closed[n]:
                result.warnings.append(f"node {n!r} is not a closed volume (holes or inconsistent normals); "
                                       "it is drawn but left out of the mass-property check")

        # 5. Mass properties against the satellite --------------------------
        sat_ids = _satellite_ids(satellite)
        declared = None
        if satellite is not None:
            declared = {"mass_kg": float(satellite.mass), "com_body_m": np.asarray(satellite.COM, float),
                        "inertia_com_kgm2": np.asarray(satellite.J_COM, float)}
            # Units sanity: size implied by the declared inertia vs the model's size.
            size_est = np.sqrt(2.0 * np.trace(declared["inertia_com_kgm2"]) / declared["mass_kg"])
            ratio = float(extent.max() / size_est)
            if ratio > 10.0 or ratio < 0.1:
                guess = min(UNIT_SCALE, key=lambda u: abs(np.log10(ratio * UNIT_SCALE[u] / scale)))
                result.warnings.append(
                    f"model is {ratio:.3g}x the size implied by the satellite's inertia "
                    f"({extent.max():.3g} m vs ~{size_est:.3g} m); units may be wrong (try \"units\": \"{guess}\")")
        mass = config.get("mass_kg", declared["mass_kg"] if declared else None)
        computed = None
        vols = {n: meshes[n] for n in meshes if closed[n]}
        if vols and (mass is not None or "density_kg_m3" in config):
            V = np.array([m.volume for m in vols.values()])
            if mass is not None:
                rho = float(mass) / V.sum()
            else:
                rho = float(config["density_kg_m3"])
                mass = rho * V.sum()
            coms = np.array([m.center_mass for m in vols.values()])
            com = (V[:, None] * coms).sum(0) / V.sum()
            inertia = np.zeros((3, 3))
            for m, v, c in zip(vols.values(), V, coms):
                inertia += _parallel_axis(rho * m.moment_inertia, rho * v, c - com)  # moment_inertia at density 1
            computed = {"mass_kg": float(mass), "volume_m3": float(V.sum()), "density_kg_m3": rho,
                        "com_body_m": com, "inertia_com_kgm2": inertia,
                        "uniform_density": True, "nodes_used": list(vols)}
        comparison = {}
        if computed and declared:
            d_com = float(np.linalg.norm(computed["com_body_m"] - declared["com_body_m"]))
            tol_com = max(COM_TOL_M, COM_TOL_FRAC_DIAG * diag)
            Jm, Jd = computed["inertia_com_kgm2"], declared["inertia_com_kgm2"]
            rel = float(np.linalg.norm(Jm - Jd) / np.linalg.norm(Jd))
            comparison = {"com_offset_m": d_com, "com_tolerance_m": tol_com, "inertia_rel_error": rel}
            if d_com > tol_com:
                result.warnings.append(f"centre of mass differs from the satellite's by {d_com * 100:.2f} cm "
                                       f"(tolerance {tol_com * 100:.2f} cm)")
            if rel > INERTIA_REL_TOL:
                result.warnings.append(f"inertia differs from the satellite's by {rel:.1%} (Frobenius; "
                                       f"tolerance {INERTIA_REL_TOL:.0%}), assuming uniform density")
            wd, vd = np.linalg.eigh(Jd)
            wm, vm = np.linalg.eigh(Jm)
            angles = []
            for i in range(3):
                gaps = np.abs(wd[i] - np.delete(wd, i)) / wd.max()
                if gaps.min() < 0.05:
                    continue  # degenerate principal axis: direction undefined
                angles.append(min(_angle_deg(vd[:, i], vm[:, j]) for j in range(3)))
            if angles:
                comparison["principal_axis_err_deg"] = max(angles)
                if max(angles) > PRINCIPAL_AXIS_TOL_DEG:
                    result.warnings.append(f"principal axes differ by up to {max(angles):.1f} deg "
                                           f"(tolerance {PRINCIPAL_AXIS_TOL_DEG:.0f} deg)")

        # 6-7. Actuator axes and sensor boresights ---------------------------
        nodes_out: Dict[str, Dict[str, Any]] = {}
        for n, m in meshes.items():
            c = node_cfg.get(n, {})
            entry = {"role": c.get("role", "body"), "triangles": int(len(m.faces)), "closed_volume": closed[n]}
            for key in ("actuator", "sensor", "articulation", "axis_body"):
                if key in c:
                    entry[key] = c[key]
            if "actuator" in c:
                act = sat_ids.get("actuators", {}).get(c["actuator"])
                if satellite is not None and act is None:
                    result.errors.append(f"node {n!r}: satellite has no actuator {c['actuator']!r}; "
                                         f"have {list(sat_ids.get('actuators', {}))}")
                elif act is not None and entry["role"] == "rw":
                    if closed[n]:
                        axis = _symmetry_axis(m.moment_inertia)
                        err = _angle_deg(axis, np.asarray(act.axis, float))
                        entry["axis_err_deg"] = err
                        if err > RW_AXIS_TOL_DEG:
                            result.warnings.append(f"node {n!r}: wheel symmetry axis is {err:.1f} deg from "
                                                   f"{c['actuator']}'s axis (tolerance {RW_AXIS_TOL_DEG:.0f} deg)")
                    entry.setdefault("axis_body", np.asarray(act.axis, float))
                    entry.setdefault("articulation", "spin")
            if "sensor" in c:
                sens = sat_ids.get("sensors", {}).get(c["sensor"])
                if satellite is not None and sens is None:
                    result.errors.append(f"node {n!r}: satellite has no sensor {c['sensor']!r}; "
                                         f"have {list(sat_ids.get('sensors', {}))}")
                elif sens is not None and sens.get("boresight_body") is not None:
                    T, _ = scene.graph[n]
                    local = np.asarray(c.get("boresight_local", [0.0, 0.0, 1.0]), float)
                    in_body = c2b[:3, :3] @ (T[:3, :3] / np.linalg.norm(T[:3, :3], axis=0)) @ local
                    err = _angle_deg(in_body, sens["boresight_body"], sign_free=False)
                    entry["boresight_err_deg"] = err
                    if err > BORESIGHT_TOL_DEG:
                        result.warnings.append(f"node {n!r}: local {local.tolist()} points {err:.1f} deg from "
                                               f"{c['sensor']}'s boresight (tolerance {BORESIGHT_TOL_DEG:.0f} deg)")
            nodes_out[n] = entry

        # 8. Drag/SRP panels vs the mesh surface ------------------------------
        panels = []
        if satellite is not None:
            from ADCS.viz.scene import build_scene

            combined = trimesh.util.concatenate(list(meshes.values()))
            fn, fc, fa = combined.face_normals, combined.triangles_center, combined.area_faces
            for i, p in enumerate(build_scene(satellite)["panels"]):
                nrm = np.asarray(p["normal_body"], float)
                nrm /= np.linalg.norm(nrm)
                facing = (fn @ nrm) > np.cos(np.radians(10.0))
                on_plane = np.abs((fc - p["center_body_m"]) @ nrm) < max(0.01, 0.02 * diag)
                area = float(fa[facing & on_plane].sum())
                rel = abs(area - p["area_m2"]) / p["area_m2"]
                panels.append({"index": i, "declared_area_m2": p["area_m2"], "mesh_area_m2": area, "rel_error": rel})
                if rel > PANEL_AREA_TOL:
                    result.warnings.append(f"panel {i} (normal {np.round(nrm, 3).tolist()}): mesh area "
                                           f"{area:.4g} m^2 vs {p['area_m2']:.4g} m^2 in the drag/SRP model")

        if result.errors:
            raise _Abort(result.errors.pop())

        # 9-10. Levels of detail and export ------------------------------------
        lods = list(config.get("lods", DEFAULT_LODS))
        max_tri = int(config.get("max_triangles", DEFAULT_MAX_TRIANGLES))
        total = sum(len(m.faces) for m in meshes.values())
        if total > max_tri:
            result.warnings.append(f"model has {total} triangles (budget {max_tri}); "
                                   "the viewer will use reduced levels of detail by default")
        lod_entries = []
        for i, ratio in enumerate(lods):
            lod_scene = trimesh.Scene()
            tris = 0
            for n, m in meshes.items():
                mm = _shade(_simplify(m, ratio))
                tris += len(mm.faces)
                lod_scene.add_geometry(mm, node_name=n, geom_name=n)
            path = out / f"{name}_lod{i}.glb"
            # Always write normals: glTF viewers (Godot included) do not
            # generate them, and a model without them only receives ambient light.
            path.write_bytes(trimesh.exchange.gltf.export_glb(lod_scene, include_normals=True))
            result.lods.append(path)
            lod_entries.append({"file": path.name, "ratio": ratio, "triangles": tris, "sha256": _sha256(path)})

        sidecar = {
            "format": "adcs-spacecraft-model", "version": 1, "name": name,
            "coords": "BODY", "units": "m",
            "note": "glb vertices are BODY-frame metres as stored (no glTF Y-up convention applied)",
            "source": {"file": str(mesh_path), "sha256": _sha256(mesh_path), "units": units,
                       "cad_to_body": c2b},
            "lods": lod_entries, "nodes": nodes_out,
            "mass_properties": {"computed": computed, "declared": declared, "comparison": comparison},
            "panels": panels,
            "report": {"errors": result.errors, "warnings": result.warnings, "info": result.info},
        }
        from ADCS.viz.writer import _jsonable

        side_path = out / f"{name}.viz.json"
        side_path.write_text(json.dumps(_jsonable(sidecar), indent=1))
        result.sidecar = side_path
    except _Abort as e:
        result.errors.append(str(e))
    return result


#: Edges sharper than this get split vertices (flat faces); smoother ones share
#: vertices (smooth shading), as CAD renderers do.
SHARP_EDGE_DEG = 30.0


def _shade(mesh: Any) -> Any:
    """Vertex normals that suit CAD geometry: hard at sharp edges, smooth elsewhere."""
    import trimesh

    if getattr(mesh.visual, "kind", None) == "texture":
        return mesh  # UV seams already define where vertices must be split
    return trimesh.graph.smooth_shade(mesh, angle=np.radians(SHARP_EDGE_DEG))


def _simplify(mesh: Any, ratio: float) -> Any:
    if ratio >= 1.0 or len(mesh.faces) < 200:
        return mesh.copy()
    import fast_simplification
    import trimesh

    v, f = fast_simplification.simplify(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32),
                                        target_reduction=1.0 - ratio)
    out = trimesh.Trimesh(vertices=v, faces=f, process=False)
    # Simplification drops texture coordinates, so reduced levels (used only
    # when the craft is small on screen) get the material's average colour.
    out.visual = trimesh.visual.ColorVisuals(out, face_colors=_mean_color(mesh))
    return out


def _mean_color(mesh: Any) -> np.ndarray:
    """RGBA uint8: base colour factor x mean texture colour, or the mesh's own colours."""
    vis = mesh.visual
    mat = getattr(vis, "material", None)
    if mat is not None:
        factor = np.asarray(getattr(mat, "baseColorFactor", None) if getattr(mat, "baseColorFactor", None) is not None
                            else [255, 255, 255, 255], dtype=float)
        factor = factor / 255.0 if factor.max() > 1.0 else factor
        tex = getattr(mat, "baseColorTexture", None)
        mean = np.ones(4)
        if tex is not None:
            arr = np.asarray(tex.convert("RGBA"), dtype=float) / 255.0
            mean = arr.reshape(-1, 4).mean(axis=0)
        return np.clip(np.rint(255.0 * factor * mean), 0, 255).astype(np.uint8)
    if getattr(vis, "kind", None) in ("face", "vertex"):
        return np.asarray(vis.face_colors, dtype=float).mean(axis=0).astype(np.uint8)
    return np.array([180, 180, 180, 255], dtype=np.uint8)


def _satellite_ids(satellite: Any) -> Dict[str, Dict[str, Any]]:
    """Actuator objects and sensor scene entries keyed by the ids used in scene.json."""
    if satellite is None:
        return {}
    from ADCS.viz.scene import _ids, build_scene

    scene = build_scene(satellite)
    return {"actuators": dict(zip(_ids(satellite.actuators), satellite.actuators)),
            "sensors": {s["id"]: {**s, "boresight_body": np.asarray(s["boresight_body"], float)
                                  if "boresight_body" in s else None} for s in scene["sensors"]}}
